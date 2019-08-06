import butterfree
from shutil import copyfile
import os
import pandas as pd

dataset_dir = butterfree.raw_dir
interim_dir = butterfree.interim_dir

def dont_preprocess(dataset):
    """ Simply copies dataset to interim folder. """
    path = os.path.join(dataset_dir, dataset, 'phenoData')
    newpath = os.path.join(interim_dir, dataset + '_phenoData.csv')
    copyfile(path, newpath)

def preprocess_all():
    for dataset in butterfree.datasets:
        if dataset in functions:
            print("Preprocessing {0}.".format(dataset))
            functions[dataset]()
        else:
            dont_preprocess(dataset)

""" These functions preprocess data from individual studies to fix formatting irregularities. """
def QinJ_2012():
    """ Instead of NaN, some of the values in this table are labelled as 'no'. """
    def f(x):
        if x == 'no':
            return float('NaN')
        else:
            return x
    path = os.path.join(dataset_dir, 'QinJ_2012', 'phenoData')
    newpath = os.path.join(interim_dir, 'QinJ_2012' + '_phenoData.csv')
    df = pd.read_csv(path)
    df['dyastolic_p'] = df['dyastolic_p'].map(f)
    df['systolic_p'] = df['systolic_p'].map(f)
    df = df.fillna('NA')
    df.to_csv(newpath, index=False)

functions = {
    'QinJ_2012': QinJ_2012,
}
