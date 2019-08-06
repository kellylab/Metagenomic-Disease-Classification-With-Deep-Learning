from fireworks import Message
import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

class ClassificationMetric(Metric):
    """
    Computes metrics involving a confustion matrics for binary classification.
    That is, metrics involving some combination of t/f positives and negatives.
    """

    def __init__(self, n, output_transform = lambda x: x):
        """ n is the number of distinct labels. """
        self.label_dim = n
        super().__init__(output_transform)

    def reset(self):
        self.tp = torch.zeros(self.label_dim)
        self.fp = torch.zeros(self.label_dim)
        self.tn = torch.zeros(self.label_dim)
        self.fn = torch.zeros(self.label_dim)
        self.num_examples = 0

    def update(self, output):

        result = output['predictions']
        predictions = torch.round(result['predictions'])
        # labels = result['label'].reshape(result['label'].shape[0],1) # Align the labels tensor with the predictions tensor.
        labels = result['label']
        tp = (predictions == 1)*(labels == 1)
        fp = (predictions == 1)*(labels == 0)
        tn = (predictions == 0)*(labels == 0)
        fn = (predictions == 0)*(labels == 1)
        self.tp += sum(tp.float()).cpu()
        self.fp += sum(fp.float()).cpu()
        self.tn += sum(tn.float()).cpu()
        self.fn += sum(fn.float()).cpu()
        self.num_examples += labels.size()[0]

    def compute(self):
        if self.num_examples == 0:
            raise NotComputableError(
                "Metric must have at least one example before it can be computed."
            )
        return self._compute()

    def _compute(self):

        # TODO: Check for divide by 0 errors
        sensitivity = self.tp / (self.tp + self.fn)
        specificity = self.tn / (self.tn + self.fp)
        ppv = self.tp / (self.tp + self.fp)
        npv = self.tn / (self.tn + self.fn)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)

        return Message({ # TODO: Map this to CPU
            'sensitivity': sensitivity.reshape(1,-1),
            'specificity': specificity.reshape(1,-1),
            'ppv': ppv.reshape(1,-1),
            'npv': npv.reshape(1,-1),
            'accuracy': accuracy.reshape(1,-1),
            'TP': self.tp.reshape(1,-1),
            'FP': self.fp.reshape(1,-1),
            'TN': self.tn.reshape(1,-1),
            'FN': self.fn.reshape(1,-1),
        })

class MulticlassClassificationMetric(ClassificationMetric):

    pass 