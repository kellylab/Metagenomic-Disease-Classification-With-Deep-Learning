# Processes raw NCBI accession code text data into python.
import pandas as pd
import re
import os

def clean(string):
    """ Removes " and spaces in string. """
    return re.sub('["\s\n]', '', string)

def isName(string):
    """ Checks to see if the string is an author name of an experiment rather than a sequence of NCBI codes. """
    return '_' in string # All of the experiment names are of the format LastFI_date, so just check for the underscore.

def extract_labels(filename):
    """ Extracts the NCBI labels in source file into a dictionary associating experiment names with labels. """
    labels = dict()
    current_name = ''
    current_labels = []

    with open(filename,'r') as NCBI:
        for line in NCBI.readlines():
            if isName(line):
                labels[current_name] = current_labels
                current_labels = []
                current_name = clean(line)
            else:
                line_labels = line.split(' ')
                line_labels = map(clean, line_labels)
                current_labels.extend(line_labels)

    labels.pop('')
    all_labels = []

    for key in labels: # Remove duplicates
        labels[key] = set(labels[key])
        all_labels.extend(labels[key])

    all_labels = set(all_labels)

    return labels, all_labels

def save_all_labels(filename, all_labels):
    """ Saves list of all labels to file. """
    with open(filename, 'w') as processed:
        processed.write('\n'.join(all_labels))

def save_labels_by_experiment(filename, labels):
    """ Saves a CSV file of NCBI codes by experiment. """
    df = pd.DataFrame.from_dict(labels, orient='index').transpose()
    df.to_csv(outfile2)

if __name__ == '__main__':

    basepath = '/home/saad/Projects/microbialcommunityprofiling' # TODO: make this relative path
    outfile = os.path.join(basepath, 'data/processed/NCBI_labels_processed')
    filename = os.path.join(basepath, 'data/processed/NCBI_labels')
    outfile2 = os.path.join(basepath, 'data/processed/NCBI_labels_cleaned_by_experiment')
    labels, all_labels = extract_labels(filename)
    save_all_labels(outfile, all_labels)
    save_labels_by_experiment(outfile2, labels)
