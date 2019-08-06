import torch
from torch import optim
from torch import nn
from torch.nn import Module
from torch.autograd import Variable
from torch.distributions import Uniform
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Loss
import visdom
import datetime
import numpy as np
from fireworks import PyTorch_Model, Pipe
import copy
from torch_geometric.nn import GCNConv, TopKPooling, HypergraphConv, SAGEConv, DenseGCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class MetaphlanNet(PyTorch_Model):

    required_components = ['widths', 'in_column', 'out_column']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.widths)-1):
            self.components['layer{0}'.format(i)] = nn.Linear(int(self.widths[i].tolist()), int(self.widths[i+1].tolist()))           
        self.num_layers = len(self.widths)-1

    def init_default_components(self):

        self.components['elu'] = nn.ELU()
        self.components['in_column'] = 'examples'
        self.components['out_column'] = 'embeddings'
        self.components['widths'] = [12365, 6000, 2000, 1000, 100]

    def forward(self, message):

        output = message[self.in_column]
        for i in range(self.num_layers):
            layer = getattr(self, 'layer{0}'.format(i))
            output = layer(output)
            output = self.elu(output)
        message[self.out_column] = output
        return message

class DeepConvNet(PyTorch_Model):

    required_components = ['channels', 'num_nodes','in_column', 'out_column', 'edge_index']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.channels)-1):
            self.components['layer{0}'.format(i)] = GCNConv(int(self.channels[i].tolist()), int(self.channels[i+1].tolist()))
            self.components['pooling{0}'.format(i)] = TopKPooling(int(self.channels[i+1].tolist()), ratio=.5)
        self.num_layers = len(self.channels)-1

    def init_default_components(self):

        self.components['elu'] = nn.ELU()
        self.components['in_column'] = 'examples'
        self.components['out_column'] = 'embeddings'
        self.components['channels'] = [1, 128, 128, 1]

    def forward(self, message):

        batch_list = Batch.from_data_list([Data(x=x, edge_index=self.edge_index) for x in message[self.in_column]])
        edge_index = batch_list.edge_index
        output = batch_list.x.reshape(-1,1)
        for i in range(self.num_layers):
            layer = getattr(self, 'layer{0}'.format(i))
            #pooling = getattr(self, 'pooling{0}'.format(i))
            output = layer(output, edge_index)
            output = self.elu(output)
            # x, edge_index, _, _, batch = pooling(output, edge_index)
            # x = gap(x, batch).transpose(0,1)
        
        message[self.out_column] = output.reshape(len(message), int(output.shape[0]*output.shape[1]/len(message)))
        
        return message
    
class Concatenator(PyTorch_Model):

    required_components = ['in_column', 'out_column', 'concatenate_column']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze() # Prevalence should not treated as a fixed prior

    def forward(self, message):

        output = message[self.in_column]
        other = message[self.concatenate_column]
        output = torch.cat((output, other),1)
        message[self.out_column] = output
        return message

class PosteriorNet(PyTorch_Model):

    required_components = ['in_column', 'out_column']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.widths)-1):
            self.components['layer{0}'.format(i)] = nn.Linear(int(self.widths[i].tolist()), int(self.widths[i+1].tolist()))            
            self.num_layers = len(self.widths)-1
    
    def init_default_components(self):

        self.components['in_column'] = 'embeddings'
        self.components['out_column'] = 'embeddings'

    def forward(self, message):

        output = message[self.in_column]
        for i in range(self.num_layers):
            layer = getattr(self, 'layer{0}'.format(i))
            output = layer(output)
            output = self.elu(output)
        message[self.out_column] = output
        return message

class Top1Classifier(PyTorch_Model):
    """ Takes the max value in its input as the output. """
    required_components = ['in_width', 'in_column', 'out_column', 'roc_bias']

    def init_default_components(self):

        self.components['in_column'] = 'predictions'
        self.components['out_column'] = 'top_predictions'

    def forward(self, batch):

        output = batch[self.in_column]
        _, indices =  torch.max(output, 1)
    
        batch[self.out_column] = indices

        return batch

class DiseaseClassifier(PyTorch_Model):

    required_components = ['in_width', 'out_width', 'in_column', 'out_column', 'roc_bias']

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if not hasattr(self, 'roc_bias'):
            self.roc_bias = nn.Parameter(torch.zeros(self.out_width))
        self.classification_layer = nn.Linear(self.in_width, self.out_width)
        self.activation = nn.Sigmoid()
        self.roc_bias.requires_grad = False


    def init_default_components(self):

        self.components['in_column'] = 'embeddings'
        self.components['out_column'] = 'predictions'

    def forward(self, batch):

        output = batch[self.in_column]
        output = self.classification_layer(output) + self.roc_bias
        output = self.activation(output)
    
        batch[self.out_column] = output 

        return batch

class DiseaseRegressor(DiseaseClassifier):
    """ This thing is essentially the same as the classifier, except the final activation is removed so that it's a regression instead of classification. """
    def forward(self, batch):
        output = batch[self.in_column]
        output = self.classification_layer(output)
    
        batch[self.out_column] = output 

        return batch

class MulticlassDiseaseClassifier(PyTorch_Model):
    """ Simply applies a softmax activation to the input in order to select one label. """
    required_components = ['in_column', 'out_column']

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def forward(self, batch):
        output = batch[self.in_column]
        output = output / sum(output) # Normalize
        batch[self.out_column] = output 

        return batch 
    
def get_evaluator(model, loss, use_cuda = True, attach = True, environment = None):

    if torch.cuda.is_available() and use_cuda:        
        model.cuda()
        cuda = True
    else:
        cuda = False

    def evaluate_function(engine, batch):
        """ This just runs the model on the batch without computing gradients. """

        predictions = model(batch)
        mean_loss = loss(batch)

        return {'loss': mean_loss.data.cpu(), 'predictions': predictions, 'labels': batch['label']}

    engine = Engine(evaluate_function)

    return engine

class InjectPrevalence(PyTorch_Model):
    """ Generates a random prevalence prior and appends it to the input. """
    
    required_components = ['prevalence', 'prevalence_column']

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.dist = Uniform(torch.Tensor([0.]), torch.Tensor([3.]))
        self._sample = True 

    def init_default_components(self):
        self.components['mean'] = 0
        self.components['variance'] = 1

    def forward(self, message):
        
        if self._sample:
            sample = self.dist.sample(self.prevalence.shape).reshape(-1)
            exp = 1+torch.exp(self.prevalence)
            if exp.device.type == 'cuda':
                sample = sample.cuda(exp.device)
            prevalence = sample*exp
            prevalence = prevalence / sum(prevalence)
            message[self.prevalence_column] = prevalence.expand(len(message), len(prevalence))
        return message

class SKLearnPipe(Pipe):

    def __init__(self, *args, model, in_column='examples', out_column='predictions', **kwargs):
        super().__init__()
        self.model = model
        self.in_column = in_column    
        self.out_column = out_column
    
    def __call__(self, batch):
        model_input = batch[self.in_column].detach().cpu().numpy()
        output = self.model.predict_proba(model_input)
        output = np.array(output)[:,:,1].transpose()
        batch[self.out_column] = torch.Tensor(output)
        return batch

class EnsembleClassifier(PyTorch_Model):

    required_components = ['examples_column', 'rf_out_column', 'nn_out_column', 'out_column', 'rf_model', 'nn_model', 'widths', 'nonlinearity']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.widths)-1):
            self.components['layer{0}'.format(i)] = nn.Linear(int(self.widths[i].tolist()), int(self.widths[i+1].tolist()))           
        self.num_layers = len(self.widths)-1
    
    def init_default_components(self):
        self.components['rf_out_column'] = 'predictions'
        self.components['nn_out_column'] = 'predictions'
        self.components['out_column'] = 'predictions'
        self.components['examples_column'] = 'examples'
        self.components['nonlinearity'] = nn.ELU()
        self.components['softmax'] = nn.Softmax(dim=1)

    def forward(self, batch):

        rf_batch = self.rf_model(batch)[self.rf_out_column]
        nn_batch = self.nn_model(batch)[self.nn_out_column]        
        examples = batch['examples']
        # Concatenate outputs
        output = torch.cat((rf_batch, nn_batch), 1)
        for i in range(self.num_layers):
            layer = getattr(self, 'layer{0}'.format(i))
            output = layer(output)
            output = self.nonlinearity(output)
        batch[self.out_column] = self.softmax(output)
        return batch

class HarmonicConvolution(PyTorch_Model): # Use PyTorch geometric

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        if self.eigenvectors.shape[0] != self.eigenvectors.shape[1]:
            raise ValueError("Eigenvectors matrix must be square.")
        l = self.eigenvectors.shape[0]
        normal = torch.distributions.Normal(0,1)
        self.g = torch.nn.Parameter(torch.diag(normal.sample((l,))))
        self.eigenvectors.requires_grad = False
        self.g.requires_grad = True 

    def init_default_components(self):

        self.components['in_column'] = 'examples'
        self.components['out_column'] = 'embeddings'

    def forward(self, batch):

        operator = torch.matmul(self.eigenvectors, self.g) # This should be precomputed
        X = batch[self.in_column]
        X = X.reshape(*X.shape, 1)
        eigenvectors = self.eigenvectors.transpose(0, 1).expand(len(batch), *self.eigenvectors.shape)
        basis_transformed_X = torch.matmul(eigenvectors, X)
        output = torch.matmul(operator, basis_transformed_X)
        batch[self.out_column] = output.reshape(*output.shape[0:2])
        return batch

class GraphPooling(): pass # Skip for now