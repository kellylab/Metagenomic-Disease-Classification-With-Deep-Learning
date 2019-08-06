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
from butterfree.pytorch import dataset, module
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

env_name = 'paper_everything_rf'
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

    test_normalizer = Normalizer()
    test_normalizer.set_state(normalizer.get_state())
    test_normalizer.enable_inference()
    test_normalizer.disable_updates()
    test_normalizer.input = test

    rf_test_set = test_normalizer[0:len(test_normalizer)]
    rf_training_set = minibatcher[0:len(oversampler)]
    rf_test_set.cpu()
    rf_training_set.cpu()
    class_weight = [
        {0: 1-weight, 1: weight} for weight in oversample_weights.numpy()
    ]
    classifier = RandomForestClassifier(class_weight=class_weight, n_estimators=10)
    classifier.fit(rf_training_set['examples'].numpy(), rf_training_set['label'].numpy().astype(int))
    predictions = classifier.predict(rf_test_set['examples'].numpy())
    probas = np.array(classifier.predict_proba(rf_test_set['examples'].numpy()))
    print("Computing accuracy for Random Forest.")
    correct = defaultdict(lambda: 0)
    top_3 = defaultdict(lambda: 0)
    top_5 = defaultdict(lambda: 0)
    total = defaultdict(lambda: 0)
    top_5_by_label = defaultdict(lambda: [])
    for i in range(probas.shape[1]): # Shape is num_classes x num_samples x 2
        predicted_probabilities = probas[:,i,1]
        t_5 = [labels_dict.inv[x] for x in (-predicted_probabilities).argsort()[:5]]
        t_3 = t_5[:3]
        t_pick = t_3[0]
        label = rf_test_set[i]['named_label'][0]
        total[label] += 1
        if label in t_pick:
            correct[label] += 1
        if label in t_3:
            top_3[label] += 1
        if label in t_5:
            top_5[label] += 1
        top_5_by_label[label].append(t_5)
    full_total = 0
    full_correct = 0
    full_top3 = 0
    full_top5 = 0
    print("Accuracy for random forest:")
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

    json.dump(top_5, experiment.open("{0}_rf_top_5.json".format(experiment_num), "w"))
    json.dump(top_3, experiment.open("{0}_rf_top_3.json".format(experiment_num), "w"))
    json.dump(correct, experiment.open("{0}_rf_correct.json".format(experiment_num), "w"))
    json.dump(total, experiment.open("{0}_rf_total.json".format(experiment_num), "w"))
    json.dump({key: value for key, value in top_5_by_label.items()}, experiment.open("rf_{0}_top_5_predictions.json".format(experiment_num), "w"))
    print("Generating ROC Curves for Random Forest.")
    fpr = {}
    tpr = {}
    thresholds = {}
    auc = {}
    for index in range(len(study_labels)):
        binary_labels = rf_test_set['label'][:, index]
        binary_predictions = predictions[:, index]
        fp, tp, th = roc_curve(binary_labels, binary_predictions)
        au = roc_auc_score(binary_labels, binary_predictions)
        fpr[labels_dict.inv[index]] = fp
        tpr[labels_dict.inv[index]] = tp
        thresholds[labels_dict.inv[index]] = th
        auc[labels_dict.inv[index]] = au
    json.dump({key: str(value) for key, value in auc.items()}, experiment.open("{0}_rf_auc.json".format(experiment_num), "w"))
    json.dump({key: str(value) for key, value in tpr.items()}, experiment.open("{0}_rf_tpr.json".format(experiment_num), "w"))
    json.dump({key: str(value) for key, value in fpr.items()}, experiment.open("{0}_rf_fpr.json".format(experiment_num), "w"))
    json.dump({key: str(value) for key, value in thresholds.items()}, experiment.open("{0}_rf_thresholds.json".format(experiment_num), "w"))
