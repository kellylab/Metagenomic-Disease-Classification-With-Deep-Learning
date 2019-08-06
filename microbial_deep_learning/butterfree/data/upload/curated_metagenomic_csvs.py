import butterfree
from caterpie import CSV, Writer
from caterpie.postgres_utils import clean
import pandas as pd
import os
import math
import numpy as np
from butterfree.data.upload.preprocess import preprocess_all

dataset_dir = butterfree.raw_dir
interim_dir = butterfree.interim_dir
datasets = butterfree.datasets

pheno_dict = {
                'sampleID': 'VARCHAR(32)',
                'subjectID': 'VARCHAR(32)',
                'antibiotics_current_use': 'BOOL',
                'study_condition': 'VARCHAR(64)',
                'disease': 'VARCHAR(400)',
                'age': 'FLOAT',
                'age_category': 'VARCHAR(20)',
                'body_site': 'VARCHAR(32)',
                'gender': 'VARCHAR(20)',
                'country': 'VARCHAR(40)',
                'non_westernized': 'BOOL',
                'DNA_extraction_kit': 'VARCHAR(30)',
                'number_reads': 'FLOAT',
                'number_bases': 'FLOAT',
                'minimum_read_length': 'FLOAT',
                'median_read_length': 'FLOAT',
                'title': 'VARCHAR(48)',
                'BMI': 'FLOAT',
                'adiponectin': 'FLOAT',
                'ajcc': 'VARCHAR(16)',
                'albumine': 'FLOAT',
                'alcohol': 'BOOL',
                'antibiotics_family': 'VARCHAR(400)',
                'bilubirin': 'FLOAT',
                'birth_control_pil': 'BOOL',
                'body_subsite': 'VARCHAR(32)',
                'c_peptide': 'FLOAT',
                'cd163': 'FLOAT',
                'cholesterol': 'FLOAT',
                'creatine': 'FLOAT',
                'creatinine': 'FLOAT',
                'ctp': 'FLOAT',
                'days_after_onset': 'FLOAT',
                'days_from_first_collection': 'FLOAT',
                'disease_subtype': 'VARCHAR(32)',
                'dyastolic_p': 'FLOAT',
                'ever_smoker': 'BOOL',
                'family': 'FLOAT',
                'ferm_milk_prod_consumer': 'VARCHAR(32)',
                'fgf_19': 'FLOAT',
                'flg_genotype': 'VARCHAR(32)',
                'fobt': 'BOOL',
                'glp_1': 'FLOAT',
                'glucose': 'FLOAT',
                'glutamate_decarboxylase_2_antibody': 'FLOAT',
                'hba1c': 'FLOAT',
                'hdl': 'FLOAT',
                'hitchip_probe_class': 'VARCHAR(32)',
                'hitchip_probe_number': 'VARCHAR(32)',
                'hla_dqa11': 'FLOAT',
                'hla_dqa12': 'FLOAT',
                'hla_drb11': 'FLOAT',
                'hla_drb12': 'FLOAT',
                'hla_dbq11': 'FLOAT',
                'hla_dbq12': 'FLOAT',
                'hscrp': 'FLOAT',
                'il_1': 'FLOAT',
                'infant_age': 'FLOAT',
                'inr':'FLOAT',
                'insulin_cat': 'BOOL',
                'lactating': 'BOOL',
                'ldl': 'FLOAT',
                'leptin': 'FLOAT',
                'location': 'VARCHAR(32)',
                'mgs_richness': 'FLOAT',
                'momeducat': 'FLOAT',
                'mumps': 'BOOL',
                'pregnant': 'BOOL',
                'protein_intake': 'FLOAT',
                'prothrombin_time': 'FLOAT',
                'shigatoxin_2_elisa': 'VARCHAR(32)',
                'smoker': 'BOOL',
                'start_solidfood': 'FLOAT',
                'stec_count': 'VARCHAR(32)',
                'stool_texture': 'VARCHAR(32)',
                'systolic_p': 'FLOAT',
                'tnm': 'VARCHAR(32)',
                'triglycerides': 'FLOAT',
                'visit_number': 'FLOAT',
                'population': 'VARCHAR(48)',
                'travel_destination': 'VARCHAR(48)',
                'lifestyle': 'VARCHAR(48)',
                'sequencing_platform': 'VARCHAR(48)',
                'PMID': 'VARCHAR(48)',
                'fasting_nsulin': 'FLOAT',
                'family_role': 'VARCHAR(24)',
                'born_method': 'VARCHAR(24)',
                'premature': 'BOOL',
                'birth_order': 'FLOAT',
                'age_twins_started_to_live_apart': 'FLOAT',
                'feeding_practice': 'VARCHAR(32)',
                'breastfeeding_duration': 'INT',
                'formula_first_day': 'INT',
                'ESR': 'FLOAT',
                'HLA': 'VARCHAR(48)',
                'autoantibody_positive': 'VARCHAR(48)',
                'age_seroconversion': 'FLOAT',
                'age_T1D_diagnosis': 'FLOAT',
                'disease_stage': 'INT',
                'disease_location': 'VARCHAR(48)',
                'calprotectin': 'FLOAT',
                'treatment': 'VARCHAR(48)',
                'remission': 'BOOL',
                'wbc': 'FLOAT',
                'rbc': 'FLOAT',
                'blood_platelet': 'FLOAT',
                'hemoglobinometry': 'FLOAT',
                'ast': 'FLOAT',
                'alt': 'FLOAT',
                'globulin': 'FLOAT',
                'urea_nitrogen': 'FLOAT',
                'ASO': 'FLOAT',
                'anti_ccp_antibody': 'FLOAT',
                'rheumatoid_factor': 'FLOAT',
                'dental_sample_type': 'VARCHAR(32)',
                'zigosity': 'VARCHAR(32)',
                'menopausal_status': 'VARCHAR(32)',
                'BASDAI': 'FLOAT',
                'BASFI': 'FLOAT',
                'HBI': 'FLOAT',
                'SCCAI': 'FLOAT',
                'birth_weight': 'FLOAT',
                'gestational_age': 'FLOAT',
                'curator': 'VARCHAR(256)',
                'uncurated_metadata': 'VARCHAR(256)',
            }

download_dict = {
                'sampleID': 'VARCHAR(32)',
                'title': 'VARCHAR(40)',
                'NCBI_accession': 'VARCHAR(10)'
                }


class PhenoCSV(CSV):
    """ Base class for phenoData csvs from curatedMetagenomicData. """

    def __init__(self, df, title):

        self.load(df)
        self.title = clean(title) # Title of the paper/study

    def preprocess(self):
        """ Formats df for uploading to postgres and generates types_dict. """
        # Convert index to a column
        self.message = self.message.rename(columns = {'Unnamed: 0': 'sampleID', 'Unnamed: 0.1': 'sampleID'})
        # Add title of study to df and remove illegal characters
        self.message['title'] = clean(self.title) # Remove illegal postgres characters.
        cleaned = {column:clean(column) for column in self.message.columns}
        self.message = self.message.rename(columns = cleaned)
        # Construct types dict
        self.construct_types_dict()
        print("Uploading {0} to table {1}.".format(self.title, self.table_name))

class PhenotypeCSV(PhenoCSV):
    """ Stores phenotypes data for studies. """

    def __init__(self, df, title):
        super().__init__(df, title)
        self.table_name = 'phenotypes'

    def construct_types_dict(self):
        self.message = self.message.drop(columns = 'NCBI_accession')
        self.types_dict = {column: pheno_dict[column] for column in self.message}
        #self.cast_types()

    def cast_types(self):
        """ Casts types in df to defined type in postgres. """
        for key in self.types_dict:
            if self.types_dict[key] in ['INT', 'BIGINT']:
                self.message[key] = [int(x) if not math.isnan(x) else x for x in self.message[key]]

class DownloadsCSV(PhenoCSV):
    """ Stores NCBI accession numbers for studies. """

    def __init__(self, df, title):

        super().__init__(df, title)
        self.table_name = 'accession'

    def construct_types_dict(self):
        """ Removes all columns except for title, id, and NCBI number. """
        self.message = self.message[list(download_dict.keys())]
        self.duplicate_rows()
        self.types_dict = {column: download_dict[column] for column in self.message}

    def duplicate_rows(self):
        """ Makes a separate row for each NCBI number. Otherwise, the NCBI numbers are provided as a list for each id. """
        df = pd.DataFrame(columns = self.message.columns)
        for i in range(len(self.message)):
            try:
                ncbi = self.message['NCBI_accession'][i].split(';')
                n = len(ncbi)
                sampleID = [self.message['sampleID'][i] for _ in range(n)]
                title = [self.message['title'][i] for _ in range(n)]
                duplicated_df = pd.DataFrame({'NCBI_accession': ncbi, 'sampleID': sampleID, 'title': title})
                df = df.append(duplicated_df)
            except:
                pass

        self.message = df

class AnnotationCSV(CSV): # NOTE: This doesn't work because postgres has a 1600 column limit
    """ Stores annotation information. """
    def __init__(self, path, title, body_site, annotation):
        #self.load(df)
        self.title = clean(title)
        self.body_site = body_site
        self.table_name = annotation
        self.path = path

    def preprocess(self):
        self.load(self.path, index_col = 0) # Only load df right before writing
        self.message = self.message.transpose()
        self.types_dict = {key: 'FLOAT' for key in self.message}
        self.message['sampleID'] = self.message.index
        self.message['title'] = self.title
        self.message['body_site'] = self.body_site
        self.types_dict['sampleID'] = 'VARCHAR(32)'
        self.types_dict['title'] = 'VARCHAR(48)'
        self.types_dict['body_site'] = 'VARCHAR(32)'

    def postprocess(self):
        self.unload()

class TrackerCSV(CSV):
    """ Stores information about where on disk annotation files are stored. """
    def __init__(self):
        self.table_name = 'annotations'

    def preprocess(self):

        annotations = [ 'genefamilies_relab',
                        'marker_abundance',
                        'marker_presence',
                        'metaphlan_bugs_list',
                        'pathabundance_relab',
                        'pathcoverage'
                        ]

        args = {'path': [], 'title': [], 'body_site': [], 'annotation': []}
        for dataset in datasets:
            dataset_dir = os.path.join(butterfree.raw_dir, dataset)
            body_sites = [x for x in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,x))]
            for body_site in body_sites:
                for annotation in annotations:
                    args['path'].append(os.path.join(dataset, body_site, annotation))
                    args['title'].append(dataset)
                    args['body_site'].append(body_site)
                    args['annotation'].append(annotation)
        self.message = pd.DataFrame(args)
        self.types_dict = {
                            'path': 'VARCHAR(256)',
                            'title': 'VARCHAR(48)',
                            'body_site': 'VARCHAR(32)',
                            'annotation': 'VARCHAR(32)'
                        }
