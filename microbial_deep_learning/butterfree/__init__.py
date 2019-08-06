import psycopg2
import configparser
import os

def save_config(filename = 'config.ini'):

    config = configparser.ConfigParser()
    config['Settings'] = {}

    ready = False;
    while not ready: #NOTE: the directory chosen here is assumed to exist and have write permissions
        path = input("Where do you want to save raw data from curatedMetagenomicData? (This is about ~20Gb) (Default: ../data/raw/): ")
        if not path or path == '':
            base_path = '/'.join(os.getcwd().split('/')[:-1]) # Go up one directory
            path = base_path + '/data/raw/'

        ready = yes_or_no("Use %s? (y or n): " % path)

    config['Settings']['raw_path'] = path

    ready = False
    while not ready:
        path = input("Where do you want to save processed data? (~Mb) (Default: ../data/): ")

        if not path or path == '':
            base_path = '/'.join(os.getcwd().split('/')[:-1]) # Go up one directory
            path = base_path + '/butterfree/data/'

        ready = yes_or_no("Use %s? (y or n): " % path)

    for sub_path in ['interim/', 'processed/', 'external/', 'experiment/']:
        try: # NOTE: these directories are assumed to exist and have write permissions.
            os.mkdir(path + sub_path)
        except: # NOTE: os.mkdir throws error if directory already exists. Hence, we are assuming that errors are because of that.
            pass

    config['Settings']['interim_path'] = path + 'interim/'
    config['Settings']['processed_path'] = path + 'processed/'
    config['Settings']['external_path'] = path + 'external/'
    config['Settings']['experiment_path'] = path + 'external/'
    config['Settings']['test_path'] = path + 'test/'

    with open(filename,'w') as configfile:
        config.write(configfile)
    print("Wrote configuration to %s" % filename)
    print("You may have to restart your python session for these changes to take effect.")

def load_config(filename = 'config.ini'):
    config = configparser.ConfigParser()
    config.read(filename)

    return config

def yes_or_no(question):
    while True:
        confirm = input(question)
        if confirm in ['yes', 'Yes', 'YES', 'fckn hell ya m8', 'y', 'Y']:
            return True
        elif confirm in ['no', 'No', 'NO', 'n', 'N']:
            return False
        else:
            print("Please respond yes or no.")

datasets = [
    'AsnicarF_2017',
    'BackhedF_2015',
    'Bengtsson-PalmeJ_2015',
    'BritoIL_2016',
    'Castro-NallarE_2015',
    'ChengpingW_2017',
    'ChngKR_2016',
    'CosteaPI_2017',
    'DavidLA_2015',
    'FengQ_2015',
    'FerrettiP_2018',
    'HanniganGD_2017',
    'Heitz-BuschartA_2016',
    'HMP_2012',
    'KarlssonFH_2013',
    'KosticAD_2015',
    'LeChatelierE_2013',
    'LiJ_2014',
    'LiJ_2017',
    'LiSS_2016',
    'LiuW_2016',
    'LomanNJ_2013',
    'LoombaR_2017',
    'LouisS_2016',
    'NielsenHB_2014',
    'Obregon-TitoAJ_2015',    
    'OhJ_2014',
    'OlmMR_2017',
    'PasolliE_2018',
    'QinJ_2012',
    'QinN_2014',
    'RampelliS_2015',
    'RaymondF_2016',
    'SchirmerM_2016',
    'ShiB_2015',
    'SmitsSA_2017',
    'TettAJ_2016',
    'ThomasAM_2018a',
    'VatanenT_2016',
    'VincentC_2016',
    'VogtmannE_2016',
    'WenC_2017',
    'XieH_2016',
    'YuJ_2015',
    'ZellerG_2014'
    ]


host = ''
dbname = ''
username = ''
password = ''
port = 0

conn_string = "user=%s password=%s dbname=%s host=%s port = %s" % (username, password, dbname, host, port)

def get_connection():
    return psycopg2.connect(conn_string)

try:
    config = load_config()
    raw_dir= config['Settings']['raw_path']
    test_dir = config['Settings']['test_path']
    processed_dir = config['Settings']['processed_path']
    interim_dir = config['Settings']['interim_path']
    external_dir = config['Settings']['external_path']
    experiment_dir = config['Settings']['experiment_path']

except:
    print("Could not find valid configuration file in config.ini. Generating new one.")
    save_config()
