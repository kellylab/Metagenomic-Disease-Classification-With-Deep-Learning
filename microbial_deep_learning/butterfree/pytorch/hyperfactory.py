from fireworks import Message
from fireworks.extensions.factory import Factory
from fireworks.utils.exceptions import EndHyperparameterOptimization
import torch
from torch.nn import Parameter
from torch.autograd import Variable
from fireworks.toolbox import BatchingPipe
import copy
from butterfree.pytorch import module
from itertools import product


class Sweeper:
    def __init__(self, min: int, max: int, dx: float, length: int):
        """
        Sweep through the range of thresholds and then end.
        Note that we do not need to test every single combination of thresholds for each label because the labels are trained
        independently.
        """
        self.min = min
        self.max = max
        self.dx = dx
        self.length = length
        self.threshold = self.min

    def __call__(self, params: list, metrics: list):

        if self.threshold >= self.max:
            raise EndHyperparameterOptimization

        else:
            #print("Threshold: {}".format(self.threshold))
            tensor = {'roc_bias': torch.FloatTensor([self.threshold for _ in range(self.length)]).reshape(1,-1)}
            self.threshold += self.dx
            return Message(tensor)

def roc_bias_generator(classifier, min=-10, max=10):
    """
    Returns a function that generates bias terms for a classifier in order to attain different sensitivity/specificity scores.
    """

    roc_bias = classifier.roc_bias
    length = len(roc_bias)
    min = min
    max = max
    dx = .4

    return Sweeper(min, max, dx, length)

def roc_trainer(classifier, loss, *args, **kwargs): # TODO: Make copying more robust
    """
    Returns a function that can retrain the classifier given a set of hyperparameters and return an evaluator.
    """
    
    def trainer(parameters):

        # classifier_copy = copy.deepcopy(classifier)
        classifier_copy = classifier.__class__(*args, **kwargs)
        classifier_copy.set_state(classifier.get_state(), reset=False)        
        if type(parameters) is Message and 'roc_bias' in parameters and parameters['roc_bias'] is not None:
            classifier_copy.roc_bias = Parameter(parameters['roc_bias'].cuda()) # TODO: Infer type
        eval_engine = module.get_evaluator(classifier_copy, loss, attach=False)
        return eval_engine

    return trainer

class ParameterSweeper:
    """
    This class will generate all permutations of those parameters.
    """
    def __init__(self, params_dict: dict):
        """
        Initialize with a dict mapping variable names to a list of values to permute over.
        """
        self.params_dict = params_dict
        self.params = product(*self.params_dict.values())

    def reset(self):
        """ Resets params_dict to restart iteration. """
        self.params = product(*self.params_dict.values())

    def __call__(self, params: list, metrics: list):
        try:
            values = self.params.__next__()
            return {key: value for key, value in zip(self.params_dict.keys(), values)}
        except StopIteration:
            raise EndHyperparameterOptimization

def trainer_from_params(train_dataset, eval_dataset, test_dataset = None):

    def trainer(parameters: dict, max_epochs: int = 15):

        converted_params = convert_keys_to_variables(parameters)
        embedder = module.MetaphlanNet(converted_params['widths'])
        classifier = module.DiseaseClassifier(converted_params['in_width'], converted_params['out_width'])
        learning_rate = converted_params['learning_rate']
        engine = module.get_trainer(embedder, classifier)
        train_loader = BatchingPipe(inputs=train_dataset)
        engine.run(train_loader, max_epochs=max_epochs)
        eval_engine = module.get_evaluator(embedder, classifier, attach=False)
        engine = None
        return eval_engine

    return trainer