import fireworks
from fireworks import Message 
import pandas as pd
import torch
import butterfree
from butterfree.data import loader
from butterfree.data.loader import get_unique_phenotypes
from butterfree.pytorch import dataset
import numpy as np
import torch
from torch.autograd import Variable
from butterfree.test.test_examples_labels import get_test_examples
from itertools import count
import os

keys = ['body_site', 'title', 'path']
query = "SELECT {0} FROM annotations where annotation = 'metaphlan_bugs_list';".format(', '.join(keys))
diseases = loader.get_unique_phenotypes('disease')
filters_dict = {disease:"SELECT sampleID, title, body_site FROM phenotypes WHERE disease LIKE '%{0}%'".format(disease) for disease in diseases}

def test_OverSampler():

    sample = Message({
        'example': [i for i in range(20)],
        'label': [*(['a' for _ in range(10)]+['b' for _ in range(6)] + ['c' for _ in range(3)] + ['d' for _ in range(1)])]
    })
    oversampler = dataset.OverSampler(input=sample)
    batch_size = 2    
    for i in range(10):
        oversampler.update(oversampler[batch_size*i:batch_size*(i+1)])
    oversampler.compile()
    counts = {'a':0, 'b':0, 'c': 0, 'd': 0}
    for row, i in zip(oversampler, count()):
        counts[row['label'][0]] += 1
    assert i == 39
    for label in ['a','b','c','d']:
        assert counts[label] == 10
    

def test_index_to_vector():
    n = 10
    index = [2]
    vector = dataset.index_to_vector(index, n)
    assert (vector == np.array([0,0,1,0,0,0,0,0,0,0])).all()
    index = [2, 5, 7]
    vector = dataset.index_to_vector(index, n)
    assert (vector == np.array([0,0,1,0,0,1,0,1,0,0])).all()

    index = [-1]
    vector = dataset.index_to_vector(index, n)
    assert (vector == np.zeros(n)).all()

def test_indices_to_vectors():

    indices = [2, [4,5], 3, 2, 10, [0, -1]]
    embeddings = dataset.indices_to_vectors(indices)
    assert (embeddings[0] == np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])).all()
    assert (embeddings[1] == np.array([0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.])).all()
    assert (embeddings[2] == np.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])).all()
    assert (embeddings[3] == np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])).all()
    assert (embeddings[4] == np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])).all()
    assert (embeddings[5] == np.zeros(11)).all()

def test_labels_to_indices():

    labels = ['IBD',
             'healthy',
             'IGT',
             'healthy',
             'HBV',
             'fatty_liver',
             'healthy;HCV;fatty_liver',
             'healthy',
             'T2D;beetus',
             'STEC',
             'healthy',
             'healthy',
             'CRC',
             'healthy',
             'healthy',
             'CD',
             'adenoma',
             'healthy',
             'cirrhosis; HCV',
             'HCV',
             ]

    # Test without using a separator
    labels_dict, indexed_labels = dataset.labels_to_indices(labels)
    assert len(set(labels)) == len(labels_dict)
    assert len(indexed_labels) == len(labels)
    for label in indexed_labels:
        assert label >= 0 and label < len(labels_dict)

    # Test while using a separator
    labels_dict, indexed_labels = dataset.labels_to_indices(labels, separator=';')
    assert 'HCV' in labels_dict
    assert 'beetus' in labels_dict
    assert len(labels_dict) == 14
    for label in indexed_labels:
        if type(label) is int:
            assert label >= 0 and label < len(labels_dict)
        else:
            for l in label:
                assert l >= 0 and l < len(labels_dict)
    assert len(indexed_labels[6]) == 3
    assert len(indexed_labels[8]) == 2
    assert type(indexed_labels[10]) is int

    # Test with a default label
    default_label = 'healthy'
    labels_dict, indexed_labels = dataset.labels_to_indices(labels, separator=';', default_label=default_label)
    assert 'HCV' in labels_dict
    assert 'beetus' in labels_dict
    assert len(labels_dict) == 14
    assert len(indexed_labels[6]) == 3
    assert len(indexed_labels[8]) == 2
    assert type(indexed_labels[10]) is int
    for label, i in zip(indexed_labels, count()):
        if type(label) is int:
            if i not in [1, 3, 6, 7, 10, 11, 13, 14, 17]:
                assert label >= 0 and label < len(labels_dict)
            else:
                assert label == -1
        else: # Is an iterable
            for l in label:
                assert l >= -1 and l < len(labels_dict)


def test_labels_to_vector():

    labels = ['IBD',
             'healthy',
             'IGT',
             'healthy',
             'HBV',
             'fatty_liver',
             'healthy;HCV;fatty_liver',
             'healthy',
             'T2D;beetus',
             'STEC',
             'healthy',
             'healthy',
             'CRC',
             'healthy',
             'healthy',
             'CD',
             'adenoma',
             'healthy',
             'cirrhosis; HCV',
             'HCV',
             ]

    labels_dict, indexed_labels = dataset.labels_to_indices(labels, separator=';')
    embeddings = dataset.indices_to_vectors(indexed_labels)
    assert type(embeddings) is list
    for embedding in embeddings:
        assert len(embedding) == 14
        assert type(embedding) is np.ndarray

    # Test with default label
    default_label = 'healthy'
    labels_dict, indexed_labels = dataset.labels_to_indices(labels, separator=';', default_label = default_label)
    embeddings = dataset.indices_to_vectors(indexed_labels)
    assert type(embeddings) is list
    for embedding in embeddings:
        assert len(embedding) == 13
        assert type(embedding) is np.ndarray
    for i in [1, 3, 6, 7, 10, 11, 13, 14, 17]:
        assert (embeddings[i] == np.zeros(13)).all()


def test_LabelEmbeddingPipe():

    ladder = loader.LoaderPipe()
    ladder.load(query, filters_dict)
    embedder = dataset.LabelEmbeddingPipe(ladder, labels_column='label')
    embedder.compute_embeddings()
    batch = embedder[0:10]
    assert batch['label'][0].shape[0] == 49
    defaultembedder = dataset.LabelEmbeddingPipe(ladder, labels_column='label', default_label='healthy')
    defaultembedder.compute_embeddings()
    batch = defaultembedder[0:10]
    assert batch['label'][0].shape[0] == 48


def test_ExampleEmbeddingPipe():

    ladder = loader.LoaderPipe()
    ladder.load(query, filters_dict)
    cols = loader.all_columns_metaphlan()
    embedder = dataset.ExampleEmbeddingPipe(ladder, examples_columns=cols)
    batch = embedder[0:10]


def test_DropDFPipe():
    ladder = loader.LoaderPipe()
    ladder.load(query, filters_dict)
    cols = loader.all_columns_metaphlan()
    embedder = dataset.ExampleEmbeddingPipe(ladder, examples_columns=cols)
    dropped = dataset.DropDFPipe(embedder)
    dropped[0:10]

def test_MetagenomePipe():

    ladder = loader.LoaderPipe()
    ladder.load(query, filters_dict)
    cols = loader.all_columns_metaphlan()
    exembedder = dataset.ExampleEmbeddingPipe(ladder, examples_columns=cols)
    laembedder = dataset.LabelEmbeddingPipe(exembedder, labels_column='label')
    dropped = dataset.DropDFPipe(laembedder)
    batch = dropped[20:30]
    for c in ['SampleID', 'label', 'examples']:
        assert c in batch.columns
