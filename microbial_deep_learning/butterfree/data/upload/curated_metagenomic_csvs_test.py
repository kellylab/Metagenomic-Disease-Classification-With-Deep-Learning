import butterfree
# from caterpie import CSV, Writer
import pandas as pd
import os
from butterfree.data.upload import curated_metagenomic_csvs as cmc

def test_phenotypeCSV():

    dataset = 'BritoIL_2016'
    dataset_dir = butterfree.raw_dir

    df = pd.read_csv(os.path.join(dataset_dir, dataset, 'phenoData'))
    csv = cmc.PhenotypeCSV(df, 'brito')
    assert csv.title == 'brito'
    assert csv.table_name == 'phenotypes'
    assert 'NCBI_accession' in csv.message.columns
    csv.preprocess()
    assert 'NCBI_accession' not in csv.message.columns
    assert 'NCBI_accession' not in csv.types_dict

def test_downloadsCSV():

    dataset = 'BritoIL_2016'
    dataset_dir = butterfree.raw_dir

    df = pd.read_csv(os.path.join(dataset_dir, dataset, 'phenoData'))
    csv = cmc.DownloadsCSV(df, 'brito')
    assert csv.title == 'brito'
    assert csv.table_name == 'accession'
    csv.preprocess()
    assert 'title' in csv.types_dict
    assert 'NCBI_accession' in csv.types_dict
    assert 'sampleID' in csv.types_dict

def test_annotationCSV():

    dataset = 'LiuW_2016'
    dataset_dir = butterfree.raw_dir

    #df = pd.read_csv(os.path.join(dataset_dir, dataset, 'stool/genefamilies_relab'))
    path = os.path.join(dataset_dir, dataset, 'stool/genefamilies_relab')
    csv = cmc.AnnotationCSV(path, 'liu', 'stool', 'genefamilies')
    assert csv.title == 'liu'
    assert csv.body_site == 'stool'
    assert csv.table_name == 'genefamilies'
    assert csv.path == path
    assert not hasattr(csv, 'message')
    csv.preprocess()
    assert hasattr(csv, 'message')
    keys = ['sampleID', 'title', 'body_site']
    for key in keys:
        assert key in csv.types_dict
    for key in csv.types_dict:
        if key not in keys:
            assert csv.types_dict[key] == 'FLOAT'
    csv.postprocess()
    assert csv.message is None
    assert csv.types_dict is None

def test_trackerCSV():

    tracker = cmc.TrackerCSV()
    assert tracker.table_name == 'annotations'
    tracker.preprocess()
    assert type(tracker.message) is pd.DataFrame
    keys = ['path', 'title', 'body_site', 'annotation']
    for key in keys:
        assert key in tracker.message
        assert key in tracker.types_dict

def test_upload():

    # datasets = [
    #     'AsnicarF_2017',
    #     'BritoIL_2016',
    #     'Castro-NallarE_2015',
    #     'ChngKR_2016',
    #     'FengQ_2015',
    #     'Heitz-BuschartA_2016',
    #     'HMP_2012',
    #     'KarlssonFH_2013',
    #     'LeChatelierE_2013',
    #     'LiuW_2016',
    #     'LomanNJ_2013',
    #     'NielsenHB_2014',
    #     'Obregon-TitoAJ_2015',
    #     'OhJ_2014',
    #     'QinJ_2012',
    #     'QinN_2014',
    #     'RampelliS_2015',
    #     'RaymondF_2016',
    #     'SchirmerM_2016',
    #     'TettAJ_2016',
    #     'VatanenT_2016',
    #     'VincentC_2016',
    #     'VogtmannE_2016',
    #     'XieH_2016',
    #     'YuJ_2015',
    #     'ZellerG_2014'
    #     ]
    #
    # for dataset in datasets:
    #     print('{0}'.format(dataset))
    #     cmc.upload_curated_csvs(datasets = [dataset], drop_previous=False)
    dataset = 'LomanNJ_2013'
    #cmc.upload_curated_csvs([dataset], False)
