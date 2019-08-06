import fireworks
from fireworks import Message, PyTorch_Model, Model
from fireworks.toolbox import Pipe, FunctionPipe
import butterfree
# from butterfree.data.loader import Loader
import torch
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from bidict import bidict
from collections import defaultdict
from fireworks.core import recursive 

class OverSampler(PyTorch_Model):

    def __init__(self, *args, labels_column='label', **kwargs):
        PyTorch_Model.__init__(self, *args, **kwargs)
        self.labels_column = labels_column # The label must be hashable bc they will be used as keys
        if self.input is not None: 
            self.sample_indices = Parameter(torch.Tensor([i for i in range(len(self.input))]))
        else:
            self.sample_indices = Parameter(torch.Tensor())
        self.freeze('sample_indices')
        self.enable_updates()

    def init_default_components(self):
        self.label_indices = defaultdict(lambda: [])
        self.current_index = 0

    def update(self, batch, **kwargs): 
        """
        Computes counts of each label.
        """
        for row in batch:            
            self.label_indices[row[self.labels_column][0]].append(self.current_index)
            self.current_index += 1 

    def compile(self):
        """
        Constructs a weighted sampling index.
        """
        # Determine upsampling level 
        counts = [len(x) for x in self.label_indices.values()]
        resample_count = max(counts)
        new_indices = np.concatenate([np.random.choice(indices, resample_count) for indices in self.label_indices.values()])
        np.random.shuffle(new_indices)

        self.sample_indices = Parameter(torch.Tensor(list(new_indices)))
        self.freeze('sample_indices')
        self.current_index = 0
    
    def __getitem__(self, index): 

        new_index = self.sample_indices[index].numpy().astype(int).tolist()
        
        return self.recursive_call('__getitem__', new_index)

    def __next__(self):

        if self.current_index >= len(self):
            raise StopIteration
        
        message = self[self.current_index]
        self.current_index += 1
        return message

    def __len__(self):

        return len(self.sample_indices)

    @recursive()
    def reset(self):
        self.current_index = 0

class Abundance(PyTorch_Model):

    def __init__(self, *args, labels_column='label', multiple_labels=False, separator=";", allowed_labels=None, **kwargs):
        PyTorch_Model.__init__(self, *args, **kwargs)
        self.labels_column = labels_column # The label must be hashable bc they will be used as keys
        self.multiple_labels = multiple_labels
        self.allowed_labels = allowed_labels
        self.separator = ";"
        self.freeze('counts')
        self.enable_updates()

    def init_default_components(self):
        self.counts = defaultdict(lambda: 0)

    def update(self, batch, **kwargs): 
        """
        Computes counts of each label.
        """

        for row in batch:
            if self.multiple_labels:
                labels = row[self.labels_column][0].split(self.separator)
                for label in labels:
                    if self.allowed_labels:
                        if label in self.allowed_labels:
                            self.counts[label] += 1
                    else:
                        self.counts[label] += 1
            else:
                self.counts[row[self.labels_column][0]] += 1

    def compile(self):
        """
        Constructs a weighted sampling index.
        """
        self.freeze('counts')
        
def CopyLabelPipe(target_column, new_column, *args, **kwargs):
    """
    Creates a new column in output which is a copy of target column.
    """
    def f(message):
        message[new_column] = message[target_column]
        return message

    return FunctionPipe(*args, function=f, **kwargs)


class LabelEmbeddingPipe(FunctionPipe):

    def __init__(self, input, labels_column, separator=None, default_label=None, allowed_labels=None):
        super().__init__(input, function = lambda x: x)
        self.labels_column = labels_column
        self.allowed_labels = allowed_labels
        self.default_label = default_label
        self.separator = separator
        self.compute_embeddings()
        self._function = self.apply_embeddings

    def compute_embeddings(self, labels=None):

        if labels is None:
            labels = self.find_labels()

        labels_dict, _ = labels_to_indices(labels, self.separator, default_label=self.default_label)
        self.labels_dict = labels_dict

    def find_labels(self, batch_size=1000):

        if self.allowed_labels:
            return self.allowed_labels
        labels = set()
        length = len(self.input)
        for index in range(int(length / batch_size)):
            labels = labels.union(set(self.input[index*batch_size:(index+1)*batch_size][self.labels_column]))

        index = max(0, length - batch_size)
        labels = labels.union(set(self.input[index:length][self.labels_column]))

        return labels

    def apply_embeddings(self, batch):

        indices = [self.apply_embedding(x) for x in batch[self.labels_column]]
        max_index = max(self.labels_dict.values())
        embeddings = torch.Tensor(np.stack(indices_to_vectors(indices, max_index=max_index)))
        batch[self.labels_column] = embeddings
        return batch

    def apply_embedding(self, x):
        
        if self.separator:
            separated = x.split(self.separator)
            if self.allowed_labels:
                embedding = [self.labels_dict[y] for y in separated if y in self.allowed_labels]
            else:
                embedding = [self.labels_dict[y] for y in separated]
            return embedding
        else:
            return self.labels_dict[x]

class LabelIndexingPipe(LabelEmbeddingPipe): 

    def apply_embeddings(self, batch):

        indices = [self.labels_dict[x] for x in batch[self.labels_column]]
        batch[self.labels_column] = torch.LongTensor(indices)
        return batch


class ExampleEmbeddingPipe(FunctionPipe):

    def __init__(self, input, examples_columns):
        super().__init__(input, function = lambda x: x)
        self.examples_columns = examples_columns
        self._function = self.apply_embeddings

    def apply_embeddings(self, batch):

        batch['examples'] = torch.Tensor(batch[[x for x in self.examples_columns]].df.values.astype(float))
        return batch

class DropDFPipe(FunctionPipe):

    def __init__(self, input):
        super().__init__(input, function=drop_df)

def drop_df(batch, exceptions=['SampleID']):
    """
    Drops the dataframe component and leaves only the tensor component of a message.
    """
    keep_df = batch[exceptions]
    new_batch = Message(keep_df)
    new_batch = new_batch.merge(batch.tensors())
    return new_batch


def labels_to_indices(labels, separator = None, labels_to_index = None, default_label = None):
    """
    Assigns each unique label a corresponding index. If separator is specified, splits each label based on the separator
    (so that an example can have multiple labels)
    Can provide a prespecified mapping dict via labels_to_dict.
    Default_label indicates a label that will correspond to an all 0's label vector. This lets you designate 'reference' labels. For example,
    if you consider 'healthy' to be the absence of a disease label in your dataset, then instead of having a 'healthy' flag in the label
    vector, you might want to set 'healthy' as a default_label so that anytime a label of all 0's is predicted or assigned, that is
    interpreted as 'healthy'.
    """

    indexed_labels = []
    index = 0

    if labels_to_index is not None:
        labels_dict = bidict(labels_to_index)
        update = False
    else:
        labels_dict = bidict()
        update = True

    def update_dict(label):
        """ Updates internal labels_dict which specifies which index to map a label to. """
        if update:
            nonlocal index
            if label not in labels_dict:
                labels_dict[label] = index
                index += 1

    if default_label is not None:
        labels_dict[default_label] = -1 # Indicates to assign a vector of all zeros

    for label in labels:
        if separator and separator in label:
            sep_labels = label.split(separator)
            for l in sep_labels:
                update_dict(l)
            indexed_labels.append([labels_dict[l] for l in sep_labels])
        else:
            update_dict(label)
            indexed_labels.append(labels_dict[label])

    return labels_dict, indexed_labels

def indices_to_vectors(indices, max_index = None):
    """
    Maps indices in input to binary vector encodings. If an example has more than one
    label, then there will be a 1 in each corresponding element of the encoding
    """
    if max_index is None:
        maxes = [max(item) if type(item) is list else item for item in indices]
        max_index = max(maxes)
    embeddings = np.array([index_to_vector(index, max_index+1) for index in indices])
    return embeddings

def index_to_vector(index, n):
    """
    Constructs an embedding vector for a given label. The index is a list of indices,
    and the embedding will have a 1 in the corresponding element of each of those indices.
    This way, an example can have more than 1 labels.
    """
    vec = np.zeros(n)
    # NOTE: If a negative index is present, then all other indices will be ignored.
    if (type(index) is int and index >= 0) or (hasattr(index, '__iter__') and not any(x < 0 for x in index)):
        vec[index] = 1
    return vec

def vector_to_index(vector, all_zeros=-1):
    """
    Converts a binary vector to a list of indices corresponding to the locations where the vector was one.
    """
    l = len(vector)
    integers = torch.Tensor([i+1 for i in range(l)]) # i+1 so that the zeroth element and the zeros vector below don't conflict
    zeros = torch.zeros(l)
    indices = torch.where(vector==1, integers, zeros) # replace an element with its index+1, otherwise zero
    flattenned_indices = indices.nonzero() - 1 # Remove all zero elements, and then correct for the i+1 from before
    if len(flattenned_indices) == 0:
        return torch.Tensor([[all_zeros]])
    else:
        return flattenned_indices

def vector_to_single_index(vector):
    l = len(vector)
    ones = torch.ones(l)
    zeros = torch.zeros(l)
    index = torch.where(vector==1, ones, zeros)
    index = int(index.nonzero()[0][0].numpy())
    return index 

def vectors_to_indices(vectors, all_zeros=-1):

    indices = [vector_to_index(v, all_zeros=all_zeros).numpy().tolist()[0] for v in vectors]
    return indices

def indices_to_labels(indices, labels_dict):

    inverted = labels_dict.inv
    labels = [[inverted[i] for i in index] for index in indices]
    return labels

def vectors_to_labels(vectors, labels_dict, all_zeros=-1):

    indices = vectors_to_indices(vectors, all_zeros=all_zeros)
    labels = indices_to_labels(indices, labels_dict)
    return labels

def vectors_to_single_labels(vectors, labels_dict):

    indices = [vector_to_single_index(vector) for vector in vectors]
    labels = [labels_dict.inv[i] for i in indices]
    return labels
