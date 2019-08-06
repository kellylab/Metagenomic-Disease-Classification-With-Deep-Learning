import torch
from torch import nn 
import butterfree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import visdom
from itertools import combinations
from sqlalchemy import create_engine, Column, Integer, String, Float, PickleType
import os
from fireworks import Message
from fireworks.toolbox import ShufflerPipe, BatchingPipe, TensorPipe
from fireworks.extensions.experiment import Experiment
from fireworks.extensions.factory import SQLFactory, LocalMemoryFactory
from fireworks.extensions.database import create_table, DBPipe
from fireworks.extensions.training import IgniteJunction, default_evaluation_closure
from fireworks.toolbox import FunctionPipe
from fireworks.toolbox.preprocessing import train_test_split, Normalizer
from butterfree.analysis.network import coo_tensor
from butterfree.data import loader
from butterfree.pytorch import dataset, module, StopTraining
from butterfree.pytorch.hyperfactory import roc_trainer, roc_bias_generator
from butterfree.pytorch.metrics import ClassificationMetric
from sklearn.decomposition import PCA
from deepexplain import pytorch as dx 
from itertools import count, cycle
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import feature_selection as fs
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import json 

env_name = 'paper_everything_cnn'
vis = visdom.Visdom(env=env_name)
description = """
    In this experiment, we are comparing random forests vs deep neural networks vs. graph convolutional neural networks.
    """
device = 0
print("Using device {}".format(device))
experiment = Experiment(env_name, butterfree.experiment_dir, description)              

def init_plot(label, additional_text = ''):

    opts = {
        'title': additional_text + '{0}'.format(label),
        'xtickmin': 0,
        'ytickmin': 0,
        'xlabel': 'False Positive Rate',
        'ylabel': 'True Positive Rate',
        'margintop': 30,
        'marginright': 20,
        'marginleft': 35,
        'marginbottom': 30,
    }

    return opts

def plot_roc(sensitivity, specificity, opts):

    # ax.scatter(1-specificity, sensitivity)
    data = np.array(pd.concat([1-specificity, sensitivity], axis=1))
    plot = vis.scatter(np.array(data), opts=opts)

    return plot

exact_study_labels = [
    'CRC',
    'T2D',
    'IGT',
    'psoriasis',
    'hypertension',
    'fatty_liver',
    'infectiousgastroenteritis',
    'IBD',
    'bronchitis',
    'T1D',
    'metabolic_syndrome',
    'otitis',
    'RA',
    'periodontitis',
    'healthy',
    ]

approximate_study_labels = [ # These diseases are frequently annotated with multiple labels. We will ignore the other labels and only treat them as the given disease.
    'AD',
    'CDI',
    'adenoma',
    'HBV',
]

study_labels = exact_study_labels + approximate_study_labels

for experiment_num in range(5):
    print("Downloading metadata from database.")
    keys = ['body_site', 'title', 'path']
    query = "SELECT {0} FROM annotations where annotation = 'metaphlan_bugs_list';".format(', '.join(keys))
    exact_filters_dict = {disease:"SELECT sampleID, title, body_site FROM phenotypes WHERE disease = '{0}'".format(disease) for disease in exact_study_labels}
    approximate_filters_dict = {disease:"SELECT sampleID, title, body_site FROM phenotypes WHERE disease LIKE '%{0}%'".format(disease) for disease in approximate_study_labels}
    filters_dict = {**exact_filters_dict, **approximate_filters_dict}
    cols = loader.all_columns_metaphlan()
    loaderpipe = loader.LoaderPipe(interpolate_columns=cols)
    loaderpipe.load(query, filters_dict)
    example_embedder = dataset.ExampleEmbeddingPipe(loaderpipe, examples_columns=cols)
    labels_embedder = dataset.LabelEmbeddingPipe(example_embedder, labels_column='label')
    labels_dict = labels_embedder.labels_dict 

    dropped = Message(labels_embedder[0:len(labels_embedder)].tensors())
    dropped['named_label'] = loaderpipe['label']
    dropped['SampleID'] = loaderpipe[0:len(loaderpipe)]['SampleID']
    dropped['label_index'] = torch.LongTensor([dataset.vector_to_single_index(x) for x in dropped['label']]) # For cross entropy loss for single classification
    shuffle_dropped = ShufflerPipe(dropped)
    shuffle_dropped.shuffle()
    dropped = shuffle_dropped[0:len(shuffle_dropped)]

    train, test = train_test_split(dropped, test=.3) 
    normalizer = Normalizer(input=train, keys=['examples'])
    normalizer.disable_inference()

    shuffled = ShufflerPipe(normalizer)
    oversampler = dataset.OverSampler(input=shuffled, labels_column='named_label')
    for batch in shuffled:
        oversampler.update(batch)
        normalizer.update(batch[['examples']])
    normalizer.compile()
    normalizer.enable_inference()
    normalizer.disable_updates()
    oversampler.compile()
    minibatcher = BatchingPipe(oversampler, batch_size=20)
    l = len(minibatcher)    
    oversample_weights = torch.Tensor([ 1 - 1/len(study_labels) for label in study_labels[:]]) # (1/n_i) / { (1\n_i) + (1\(n-n_i)) }
    
    def inject_oversample_weights(x):
        """ Adds a 'prevalence' column, where the prevalence values are tuned such that each individual classifier is oversampled to 50/50 """
        x['prevalence'] = oversample_weights.expand(len(x), len(oversample_weights))
        return x

    training_set = TensorPipe(FunctionPipe(minibatcher, function=inject_oversample_weights), columns=['examples', 'label', 'prevalence', 'label_index'], device=device)

    print("Initializing Model.")

    coo = torch.load('phylogenetic_tree.torch') # This is the adjacency matrix for the tree in COO format.

    conv_net = module.DeepConvNet(components={"channels": [1, 64, 64, 1], "edge_index": coo, "num_nodes": 12365})
    deep_net = module.MetaphlanNet(components={"widths": [12365, 350], "in_column": 'embeddings'}, input=conv_net)
    classifier = module.DiseaseClassifier(input=deep_net, components={'in_column': 'embeddings', 'in_width': 350, 'out_width': len(study_labels)})
    
    single_classifier = module.MulticlassDiseaseClassifier(input=classifier, components={'in_column': 'embeddings', 'out_column': 'top_prediction'})

    bce = nn.BCELoss(size_average=False)
    ce = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        conv_net.edge_index = conv_net.edge_index.cuda(device=device)
        conv_net.cuda(device=device)
        deep_net.cuda(device=device)
        classifier.cuda(device=device)
        single_classifier.cuda(device=device)
        bce.cuda(device=device)
        ce.cuda(device=device)

    def loss(batch):
        loss_multiplier = batch['prevalence']
        bce.weight = loss_multiplier[0].detach() # HACK: It should be possible to specify different weights per row
        ce.weight = loss_multiplier[0].detach()
        return 100*bce(batch['predictions'], batch['label']) 

    def training_loss(batch):
        l = loss(batch)
        if l < 200:
            raise StopTraining
        return l

    conv_trainer = IgniteJunction(components={'model': single_classifier, 'dataset': training_set}, loss=training_loss, optimizer='Adam', lr=.00002, weight_decay=1, visdom=True, environment=env_name)
    json.dump(dict(labels_dict), experiment.open("deep_{0}_cnn_labels_dict.json".format(experiment_num), "w"))

    with torch.cuda.device(device):
        print("Now training")
        training_set.shuffle()
        minibatcher.batch_size = 40
        try:
            conv_trainer.run(max_epochs=40)
        except StopTraining:
            pass

    test_normalizer = Normalizer()
    test_normalizer.set_state(normalizer.get_state())
    test_normalizer.enable_inference()
    test_normalizer.disable_updates()
    test_normalizer.input = BatchingPipe(test, batch_size=20)
    test_set = TensorPipe(FunctionPipe(ShufflerPipe(test_normalizer), function=inject_oversample_weights), columns=['examples', 'label', 'prevalence', 'label_index'], device=device)

    # Save models
    state = conv_net.get_state()
    Message.from_objects(state).to('json', path=experiment.open("conv_net.json",string_only=True))
    state = deep_net.get_state()
    Message.from_objects(state).to('json', path=experiment.open("deep_net.json",string_only=True))
    state = classifier.get_state()
    Message.from_objects(state).to('json', path=experiment.open("classifier.json",string_only=True))
    Message.from_objects(state).to('json', path=experiment.open("deep_only.json",string_only=True))
    state = None

    # Compute accuracy
    top_5_by_label = defaultdict(lambda: [])
    print("Computing accuracy for CNN")
    correct = defaultdict(lambda: 0)
    top_3 = defaultdict(lambda: 0)
    top_5 = defaultdict(lambda: 0)
    total = defaultdict(lambda: 0)
    for batch in test_set:
        batch['prevalence'] = oversample_weights.expand(len(batch), len(study_labels)).cuda()
        batch = single_classifier(batch)
        predictions = batch['predictions']
        label_indices = batch['label_index']
        
        for row, l, p in zip(batch, label_indices, predictions):
            label = row['named_label'][0]
            total[label] += 1
            #label = labels_dict.inv[int(p)]
            # Compute top 1 accuracy
            top_prediction = torch.max(p,0)[1]
            if l == top_prediction:
                correct[label] += 1
            # Compute top 3 accuracy
            top_3_predictions = torch.topk(p, 3)[1]
            if l in top_3_predictions:
                top_3[label] += 1
            top_5_predictions = torch.topk(p, 5)[1]
            top_5_by_label[label].append(top_5_predictions)
            if l in top_5_predictions:
                top_5[label] += 1

    full_total = 0
    full_correct = 0
    full_top3 = 0
    full_top5 = 0
    print("Accuracy for conv net:")
    for label in study_labels:
        if total[label] > 0:
            full_total += total[label]
            full_correct += correct[label]
            full_top3 += top_3[label]
            full_top5 += top_5[label]
            print("Accuracy on test set for {0}: {1:.2f}% out of {2} samples".format(label, float(correct[label])/float(total[label]) * 100, total[label]))
            print("Top 3 Accuracy on test set for {0}: {1:.2f}% out of {2} samples".format(label, float(top_3[label])/float(total[label]) * 100, total[label]))
            print("Top 5 Accuracy on test set for {0}: {1:.2f}% out of {2} samples".format(label, float(top_5[label])/float(total[label]) * 100, total[label]))
            print("\n")
    print("Total accuracy on test set: {0:.2f}%".format(float(full_correct)/float(full_total) * 100))
    print("Total top-3 accuracy on test set: {0:.2f}%".format(float(full_top3)/float(full_total) * 100))
    print("Total top-5 accuracy on test set: {0:.2f}%".format(float(full_top5)/float(full_total) * 100))

    json.dump(top_5, experiment.open("deep_{0}_cnn_top_5.json".format(experiment_num), "w"))
    json.dump(top_3, experiment.open("deep_{0}_cnn_top_3.json".format(experiment_num), "w"))
    json.dump(correct, experiment.open("deep_{0}_cnn_correct.json".format(experiment_num), "w"))
    json.dump(total, experiment.open("deep_{0}_cnn_total.json".format(experiment_num), "w"))
    json.dump({key: torch.stack(value).tolist() for key, value in top_5_by_label.items()}, experiment.open("deep_{0}_cnn_top_5_predictions.json".format(experiment_num), "w"))
    
    params_table = create_table('parameters', columns=[Column('roc_bias', PickleType)])
    metrics_tables = {'classification': create_table('classification', columns=[
        Column('sensitivity', Float),
        Column('specificity', Float),
        Column('TP', Float),
        Column('FP', Float),
        Column('FN', Float),
        Column('TN', Float),
        Column('npv', Float),
        Column('ppv', Float),
        Column('accuracy', Float),
        ])}
    print("Tuning ROC curves for CNN")
    with torch.cuda.device(device):
        tuning_trainer = roc_trainer(classifier, loss=lambda batch: bce(batch['predictions'],batch['label']) , input=deep_net, components={'in_width': 350, 'out_width': len(study_labels)})
        metrics = {'classification': ClassificationMetric(len(study_labels))}
        generator = roc_bias_generator(classifier, min=-50, max=50)
        factory = LocalMemoryFactory(components={
            'trainer': tuning_trainer, 
            'metrics': metrics, 
            'eval_set': test_set, 
            'parameterizer': generator, 
        })
        factory.run()
    print("Plotting ROC Curves")
    param_list, metrics_list = factory.read()
    bulk_classification = metrics_list['classification']
    classification_dict = {}
    length = len(bulk_classification['sensitivity'])
    num_labels = len(study_labels)
    for label, label_index in labels_dict.items():
        classification_dict[label] = {}
        cd = classification_dict[label]
        for key in ['sensitivity', 'specificity', 'ppv', 'npv', 'accuracy', 'TP', 'FP', 'TN', 'FN']:
            cd[key] = pd.Series(bulk_classification[key][:,label_index])
    for study_label in study_labels[:]:
        opts = init_plot(study_label, additional_text="CNN_{0}_{1}: ".format(experiment_num, device))
        plot = plot_roc(classification_dict[study_label]['sensitivity'], classification_dict[study_label]['specificity'], opts)
        vis.save([env_name])
    for key, value in classification_dict.items(): 
        Message(value).to_csv(experiment.open('{0}_cnn_{1}_metrics.csv'.format(experiment_num ,key), string_only=True))
