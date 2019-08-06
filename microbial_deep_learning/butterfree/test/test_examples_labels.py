# This script generates sample example/label pairs for unit testing.
from butterfree.pytorch import module
from butterfree.data import loader 
from butterfree.pytorch import dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pickle
import os
import numpy as np
from fireworks import Message
import logging
import pandas as pd 


def make_test_examples():

    logging.info("Downloading data from database and file system.")
    keys = ['body_site', 'title', 'path']
    query = "SELECT {0} FROM annotations where annotation = 'metaphlan_bugs_list';".format(', '.join(keys))
    diseases = loader.get_unique_phenotypes('disease')
    filters_dict = {disease: "SELECT sampleID, title, body_site FROM phenotypes WHERE disease LIKE '%{0}%'".format(disease) for disease in diseases}

    ladder = loader.LoaderPipe()
    ladder.load(query, filters_dict)
    cols = loader.all_columns_metaphlan()
    exembedder = dataset.ExampleEmbeddingPipe(ladder, examples_columns=cols)
    laembedder = dataset.LabelEmbeddingPipe(exembedder, labels_column='label')

    dropped = Message(laembedder[0:len(laembedder)].tensors())
    dropped['SampleID'] = ladder[0:len(ladder)]['SampleID']

    logging.info("Saving data")

    try:  # Remove file if it exists
        os.remove('SampleID.csv')
    except OSError:
            pass
    try:
        os.remove('examples.torch')
    except OSError:
        pass
    try:
        os.remove('label.torch')
    except OSError:
        pass

    dropped[['SampleID']].to_csv('SampleID.csv', index=False) # TODO: Make this the default way that Messages are saved.
    torch.save(dropped['examples'], 'examples.torch')
    torch.save(dropped['label'], 'label.torch')


def read_data():

    sampleids = pd.read_csv('SampleID.csv')
    examples = torch.load('examples.torch')
    label = torch.load('label.torch')
    data = Message({'examples': examples, 'label': label}, sampleids)

    return data


def get_test_examples():

    try:
        data = read_data()
    except:  # If it doesn't exist yet, make it
        make_test_examples()
        data = read_data()

    return data
