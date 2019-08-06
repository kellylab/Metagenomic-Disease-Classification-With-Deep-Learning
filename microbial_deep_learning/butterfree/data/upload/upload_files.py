import butterfree
from caterpie import CSV, Writer
from caterpie.postgres_utils import clean, drop_table
from butterfree.data.upload.curated_metagenomic_csvs import PhenotypeCSV, AnnotationCSV, DownloadsCSV, TrackerCSV
import pandas as pd
import os
import math
import numpy as np
from butterfree.data.upload.preprocess import preprocess_all
        
def upload_curated_csvs(datasets=butterfree.datasets, drop_previous=True):

    if drop_previous:
        conn = butterfree.get_connection()
        drop_table('phenotypes', conn)
        drop_table('downloads', conn)

    phenotypes = []
    ncbi = []
    preprocess_all()
    for dataset in datasets:
        df = pd.read_csv(os.path.join(butterfree.interim_dir, dataset + '_phenoData.csv'))
        phenotypes.append(PhenotypeCSV(df, dataset))
        ncbi.append(DownloadsCSV(df, dataset))
    pheno_writer = Writer(phenotypes, conn, backend='postgresql')
    pheno_writer.write_tables()
    ncbi_writer = Writer(ncbi, conn)
    ncbi_writer.write_tables()

def upload_otus(datasets=butterfree.datasets, drop_previous=True):

    # NOTE: This does not work bc postgres column limit.
    print("This does not work because of the postgres column limit. Cancelling request.")
    return
    if drop_previous:
        conn = butterfree.get_connection()
        for annotation in annotations:
            drop_table(annotation, conn)

    n = len(args['path'])
    for key in args:
        assert len(args[key]) == n
    otus = []
    for i in range(n):
        otus.append(AnnotationCSV(args['path'][i], args['title'][i], args['body_site'][i], args['annotation'][i]))
    writer = Writer(otus)
    writer.write_tables()

def upload_references(drop_previous=True):

    table_name = 'annotations'
    if drop_previous:
        conn = butterfree.get_connection()
        drop_table(table_name, conn)
    tracker = TrackerCSV()
    writer = Writer([tracker], conn)
    writer.write_tables()
    print("Wrote references to table {0}".format(table_name))
