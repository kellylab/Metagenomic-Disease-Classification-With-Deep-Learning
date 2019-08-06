import fireworks
from fireworks import Pipe, Message
from fireworks.extensions.database import TablePipe
from collections import defaultdict
import functools
import butterfree
import pandas as pd
import csv
import os
import pickle

def get_row_ids(query, filters_dict, keys=['body_site', 'title', 'path'], filter_keys=['sampleID', 'title', 'body_site']):
    """
    args:
        query: A SQL query that returns a table containing filepaths along with file ids (paper title)
        filters_dict: A dict of the form { label : q}, where the results of SQL query q will have label 'label'. Additionally, the result
            of this query should have file_id as one of the columns.
    returns:
        Returns a dict mapping { filepaths : row ids }
    """
    conn = butterfree.get_connection()
    curr = conn.cursor()
    # Execute query
    curr.execute(query)
    qf = pd.DataFrame(curr.fetchall(), columns=keys)
    # Execute filter queries
    curr.close()
    filters = pd.DataFrame(columns=filter_keys)
    for f in filters_dict.values():
        curr = conn.cursor()
        curr.execute(f)
        filters = filters.append(pd.DataFrame(curr.fetchall(), columns=filter_keys))
        curr.close()
    # Match columns
    conn.close()
    filters['path'] = ['' for _ in range(len(filters))]
    merged = pd.DataFrame(columns=['body_site', 'title', 'path', 'sampleID'])
    for _, row in qf.iterrows():
        matched = filters.loc[filters['title'] == row['title']]
        matched['path'] = row['path']
        matched['body_site'] = row['body_site']
        merged = merged.append(matched)

    row_ids = {path: [] for path in set(merged.path)}
    for _, row in merged.iterrows():
        row_ids[row['path']].append(row['sampleID'])

    return row_ids

def get_row_labels(query, filters_dict, keys=['body_site', 'title', 'path'], filter_keys=['sampleID', 'title', 'body_site']):
    """
    args:
        query: A SQL query that returns a table containing filepaths along with file ids (paper title)
        filters_dict: A dict of the form { label : q}, where the results of SQL query q will have label 'label'. Additionally, the result
            of this query should have file_id as one of the columns.
    returns:
        Returns a dict mapping { labels : row_ids }
    """
    conn = butterfree.get_connection()
    curr = conn.cursor()
    # Execute query
    curr.execute(query)
    qf = pd.DataFrame(curr.fetchall(), columns=keys)
    # Execute filter queries
    curr.close()
    labels_dict = {}
    for label, f in filters_dict.items():
        curr = conn.cursor()
        curr.execute(f)
        filters = pd.DataFrame(curr.fetchall(), columns=filter_keys)
        labels_dict[label] = filters['sampleID'].tolist()
        curr.close()

    return labels_dict

def get_file_labels(query, filters_dict, keys=['body_site', 'title', 'path'], filter_keys=['sampleID', 'title', 'body_site']):
    """
    args:
        query: A SQL query that returns a table containing filepaths along with file ids (paper title)
        filters_dict: A dict of the form { label : q}, where the results of SQL query q will have label 'label'. Additionally, the result
            of this query should have file_id as one of the columns.
    returns:
        Returns a dict mapping { filepaths : df }, where df is a DataFrame with columns row_id and labels
    """
    conn = butterfree.get_connection()
    curr = conn.cursor()
    # Execute query
    curr.execute(query)
    qf = pd.DataFrame(curr.fetchall(), columns=keys)
    # Execute filter queries
    curr.close()
    label_df = pd.DataFrame(columns=filter_keys+['label'])
    for label, f in filters_dict.items():
        curr = conn.cursor()
        curr.execute(f)
        filters = pd.DataFrame(curr.fetchall(), columns=filter_keys)
        filters['label'] = label
        label_df = label_df.append(filters)
        curr.close()

    # Get body sites and paths associated with titles
    body_sites_dict = {row['path']: row[['body_site', 'title']] for _, row in qf.iterrows()}
    labels_dict = {}
    for path, row in body_sites_dict.items():
        matched = (label_df['title'] == row['title'])*(label_df['body_site'] == row['body_site']) # Get rows where title and body site match
        if sum(matched) > 0:
            extracted = label_df[matched]
            labels_dict[path] = extracted[['sampleID', 'label']]
    return labels_dict

def extract_row_ids(filepath, row_ids):
    """
    args:
        filepath: Path to file.
        row_ids: The row_ids to search for in file.
    returns:
        Extracts and returns rows matching a row id from a given file.
    """
    df = pd.read_csv(os.path.join(butterfree.raw_dir,filepath), index_col=0).transpose()
    try:
        intersection = df.index.intersection(row_ids)
        return df.loc[row_ids]
    except:
        assert False

def interpolate(df, columns, fill_value =  0.):
    """
    Interpolates df so that it's columns are the same as the columns argument.
    If a column in df is present in columns, then it is retained.
    If a column in df is not present in columns, it is dropped.
    If a column in columns is not present in df, then that column is filled in with fill_value.
    """
    interp_df = pd.DataFrame(columns=columns) # Create an empty dataframe with the desired columns
    difference = df.columns.difference(columns) # Columns in df but not in columns will be dropped
    dropped_df = df.drop(columns=difference)
    return pd.concat([dropped_df, interp_df]).fillna(fill_value)

def save_intermediate_file(labels_dict, directory, interpolate_columns, fill_value=0):
    """
    Extracts rows matching row ids in a dict of { filepaths : row'sample_id' ids } and saves to a new file,  which
    has a table index by row id and columns that correspond to the columns in the original files.
    A hook function can be passed in to modify the dataframe before writing to disk (such as to standardize columns).

    Simultaneously saves another file that maps row id to labels.

    args:
        id_dict: Dict of the form { filepath : row_ids }
        labels_dict: Dict of the form { filepath : labels_df}
    """

    with open(directory, 'a+') as intermediate:
        header = True
        for filepath, labels_df in labels_dict.items():
            # Extract
            df = extract_row_ids(filepath, labels_df['sampleID'].tolist())
            df = interpolate(df, interpolate_columns, fill_value=fill_value)
            labels_df.index = labels_df['sampleID']
            df['label'] = labels_df['label']
            # Write to file
            df.to_csv(intermediate, header=header)
            header=False

characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9','.',';',',','%','$','_','-']

def assign_file_name(query, filters_dict):


    everything = '_'.join([str(query)] + list(filters_dict.keys()) + list(filters_dict.values()))
    counts = count_characters(everything)
    pseudo_hash = ''.join(['%s%s' % (character, counts[character]) for character in characters])
    file_name = pseudo_hash + '.csv'

    return file_name

def count_characters(string):

    return {character: string.count(character) for character in characters}

@functools.lru_cache()
def get_unique_phenotypes(column):
    """ Returns all unique values of a phenotype from a chosen column in the 'phenotypes' table. """
    conn = butterfree.get_connection()
    curr = conn.cursor()
    curr.execute("SELECT DISTINCT {0} FROM phenotypes;".format(column))
    fetched = curr.fetchall()
    result = []
    for f in fetched:
        if f[0]:
            result.extend(f[0].split(';'))
    result = list(set(result))
    return result

class LoaderPipe(Pipe):
    """
    Given a query and filter, either loads target file from disk or creates it and then loads it.
    """
    def __init__(self, input=None, directory=None, interpolate_columns=None):
        super().__init__(input=None)
        self.file_directory = directory or butterfree.interim_dir
        self.interpolate_columns = interpolate_columns
        self._current_index = 0

    def load(self, query, filters_dict, keys=None, fill_value=0):
        """
        Loads from database and filesystem records corresponding to the provided query and filter dict.
        """
        filename = assign_file_name(query, filters_dict)
        self._path = os.path.join(self.file_directory, filename)
        # Check if file already exists
        try:
            self.message = Message.read('csv', self._path)
        except:
            self._create_file(query, filters_dict, keys, fill_value)
            self.message = Message.read('csv', self._path)
        self.message['SampleID'] = self.message['Unnamed: 0']

    @property
    def filename(self):
        try:
            return self._path
        except:
            filename = assign_file_name(query, filters_dict)
            self._path = os.path.join(self.file_directory, filename)
            return self._path

    def _create_file(self, query, filters_dict, keys=None, fill_value=0):

        if keys is not None:
            labels_dict = get_file_labels(query, filters_dict, keys)
        else:
            labels_dict = get_file_labels(query, filters_dict)

        save_intermediate_file(labels_dict, self._path, self.interpolate_columns, fill_value=fill_value)
    def __getitem__(self, index):

        return self.message[index]

    def __len__(self):

        return len(self.message)

    def __iter__(self):

        self._current_index = 0
        return self

    def __next__(self):

        if self._current_index >= len(self):
            raise StopIteration

        item = self.message[self._current_index]
        self._current_index += 1

        return item

def get_title(path):
    """ Returns the title of the paper as implied by the given filepath. """
    #units = path.replace('-','_').split('/') # NOTE: - must be replaced with _ because postgres does not support - in strings.
    units = path.split('/')
    for title in butterfree.datasets:
        if title in units:
            return title

def load_annotations(paths):
    """ Returns annotations data from csvs in paths and concatenates them together to have the same index or specified index. """

    """
        if annotation is 'metaphlan':
        reference_columns = all_columns_metaphlan()
        use_reference = True
        interpolator = lambda df: interpolate(df, reference_columns, fill_value = 0.)
    """
    csvs = []
    #lengths = [0]
    df_dict = defaultdict(pd.DataFrame)
    for path in paths:
        title = get_title(path)
        # title = title.replace('-', '_') # NOTE: Test this
        df = pd.read_csv(os.path.join(butterfree.raw_dir,path), index_col=0).transpose()
        df_dict[title] = df_dict[title].append(df)
    columns = pd.Index([])
    for title, df in df_dict.items():
        columns = columns.union(df.columns)
    #df_dict = {key: interpolate(value, reference_columns) for (key, value) in df_dict.items()}

    #return dict(df_dict)
    return columns

def get_all_columns(annotation):
    conn = butterfree.get_connection()
    curr = conn.cursor()
    keys = ['body_site', 'title', 'path']
    curr.execute("SELECT {0} FROM annotations where annotation = '{1}';".format(', '.join(keys), annotation))
    references = pd.DataFrame(curr.fetchall(), columns = keys)
    columns = load_annotations(references['path'])

    return columns

def get_columns(filename, annotation, use_cache, update_cache):

    if use_cache:
        try:
            columns = pickle.load(open(filename, 'rb'))['columns']
        except FileNotFoundError:
            columns = get_all_columns(annotation)
            if update_cache:
                pickle.dump({'columns': columns}, open(filename, 'wb'))
    else:
        columns = get_all_columns(annotation)
        if update_cache:
            pickle.dump({'columns': columns}, open(filename, 'wb'))
    
    return columns
    
def all_columns_functional(use_cache = True, update_cache = True):
    """ Returns all functional categories identified in the data set. """
    return get_columns('pathabundance_columns.pickle', 'pathabundance_relab', use_cache, update_cache)

def all_columns_metaphlan(use_cache = True, update_cache = True):
    """ Returns all metaphlan OTUs identified in the data set. """
    return get_columns('metaphlan_columns.pickle', 'metaphlan_bugs_list', use_cache, update_cache)

    """ 
    def get_all_columns_metaphlan():
        conn = butterfree.get_connection()
        curr = conn.cursor()
        keys = ['body_site', 'title', 'path']
        curr.execute("SELECT {0} FROM annotations where annotation = 'metaphlan_bugs_list';".format(', '.join(keys)))
        references = pd.DataFrame(curr.fetchall(), columns = keys)
        df_dict = load_annotations(references['path'])
        merged = pd.concat([df_dict[key] for key in df_dict])
        return merged.columns

    if use_cache:
        try:
            columns = pickle.load(open('metaphlan_columns.pickle', 'rb'))['columns']
        except FileNotFoundError:
            columns = get_all_columns_metaphlan()
            if update_cache:
                pickle.dump({'columns': columns}, open('metaphlan_columns.pickle', 'wb'))
    else:
        columns = get_all_columns_metaphlan()
        if update_cache:
            pickle.dump({'columns': columns}, open('metaphlan_columns.pickle', 'wb'))

    return columns
     """