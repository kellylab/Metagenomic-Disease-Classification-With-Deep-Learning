import butterfree
from butterfree.data import loader
import pandas as pd
import cProfile
import pstats


keys = ['title', 'path']
query = "SELECT {0} FROM annotations where annotation = 'metaphlan_bugs_list';".format(', '.join(keys))
diseases = loader.get_unique_phenotypes('disease')
filters_dict = {disease:"SELECT sampleID, title FROM phenotypes WHERE disease LIKE '%{0}%'".format(disease) for disease in diseases}

llama = loader.Loader(query, keys, filters_dict)
cProfile.run('llama.get_examples_and_labels()', 'get_examples_and_labels')
p = pstats.Stats('get_examples_and_labels')
