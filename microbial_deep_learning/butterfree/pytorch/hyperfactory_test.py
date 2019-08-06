from fireworks import Message
from fireworks.toolbox import BatchingPipe
from fireworks.extensions.factory import SQLFactory, LocalMemoryFactory
import torch
from torch import nn
from butterfree.pytorch import hyperfactory, module
from butterfree.data.loader import get_unique_phenotypes
from butterfree.pytorch import dataset
from butterfree.pytorch.metrics import ClassificationMetric
from torch.utils.data import DataLoader
from butterfree.test.test_examples_labels import get_test_examples
from itertools import product
from fireworks.utils.exceptions import EndHyperparameterOptimization
import ignite
from sqlalchemy import create_engine, Column, Integer, String, Float, PickleType
from fireworks.extensions.database import create_table, DBPipe

essential_labels = [
        'asthma',
        'AD',
        'pneumonia',
        'CRC',
        'healthy',
        'pyelonefritis',
        'T2D',
        'arthritis',
        'pyelonephritis',
        'cellulitis',
        'cirrhosis',
        'IGT',
        'fatty_liver',
        'stomatitis',
        'CD',
        'adenoma',
        'psoriasis',
        'hypertension',
        'infectiousgastroenteritis',
        'tonsillitis',
        'AR',
        'IBD',
        'HBV',
        'bronchitis',
    ]

params_table = create_table('parameters', columns=[Column('roc_bias', PickleType)])
metrics_tables = {'Classification': create_table('classification', columns=[
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

def test_roc_bias_generator():

    in_width = 50
    out_width = 55
    classifier = module.DiseaseClassifier(components={'in_width': in_width, 'out_width': out_width})
    bias_generator = hyperfactory.roc_bias_generator(classifier)
    new_bias = bias_generator(None, None)
    assert type(new_bias) is Message
    assert 'roc_bias' in new_bias
    assert len(new_bias['roc_bias'][0]) == out_width
    assert len(new_bias['roc_bias']) == 1

def test_roc_trainer():

    data = get_test_examples()
    metanet = module.MetaphlanNet(components={'widths':[12365, 6000, 2000, 55]})
    classifier = module.DiseaseClassifier(components={"in_width": 55, "out_width": 49}, input=metanet)
    if torch.cuda.is_available():
        data.cuda()
        metanet.cuda()
        classifier.cuda()
    bce = nn.BCELoss()
    def loss(batch): return bce(batch['predictions'], batch['label'])
    trainer = hyperfactory.roc_trainer(classifier, loss, components={"in_width": 55, "out_width": 49}, input=metanet)
    evaluator = trainer(Message({'roc_bias': torch.ones(49)}))
    assert hasattr(evaluator, 'run')
    evaluator.run(data)


def test_ParameterSweeper():

    a = [1,2,3]
    b = ['i', 'am', 'groot']
    p = product(a, b)
    c = []
    for item in p:
        c.append(set(item))
    swipy = hyperfactory.ParameterSweeper({'hi': b, 'there': a})
    d = []
    while True:
        try:
            params = swipy(None, None)
            assert params['hi'] in b
            assert params['there'] in a
            d.append(tuple(params.values()))
        except EndHyperparameterOptimization:
            break
    for tupl in d:
        assert set(tupl) in c
