from butterfree.pytorch import module
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from butterfree.test.test_examples_labels import get_test_examples


def test_DiseaseClassifier():

    metanet = module.MetaphlanNet(components={'widths': [12365, 6000, 2000, 100]})
    classifier = module.DiseaseClassifier(components={'in_width': 100, 'out_width': 55})
    data = get_test_examples()
    if torch.cuda.is_available():
        metanet.cuda()
        classifier.cuda()
        data.cuda()
    output = classifier(metanet((data[2])))
    assert len(output['embeddings'][0]) == 100
    output = classifier(metanet(data[3:10]))
    assert output['predictions'].shape == torch.Size([7, 55])


def test_set_state_get_state():
    
    classifier = module.DiseaseClassifier(components={'in_width': 100, 'out_width': 55})
    state = classifier.get_state()
    new_classifier = module.DiseaseClassifier(components={'in_width': 100, 'out_width': 55})
    assert not (classifier.state_dict()['classification_layer.weight'] == new_classifier.state_dict()['classification_layer.weight']).all()
    new_state = new_classifier.get_state()
    new_classifier.set_state(state, reset=False)
    assert (classifier.state_dict()['classification_layer.weight'] == new_classifier.state_dict()['classification_layer.weight']).all()


def test_MetaphlanNet():

    metanet = module.MetaphlanNet(components={'widths': [12365, 6000, 2000, 100]})
    state = metanet.get_state()
    betanet = module.MetaphlanNet()
    crate = betanet.get_state()

    classifier = module.DiseaseClassifier(components={'in_width': 100, 'out_width': 55})
    data = get_test_examples()
    if torch.cuda.is_available():
        metanet.cuda()
        classifier.cuda()
        data.cuda()
    output = classifier(metanet((data[0:10])))
    assert len(output['embeddings'][0]) == 100
    output = classifier(metanet((data[0:10])))
    assert output['predictions'].shape == torch.Size([10, 55])

def test_PrevalenceNet():

    metanet = module.MetaphlanNet(components={'widths': [12365, 6000, 1000]})
    prior = torch.empty(55).uniform_(0,1)
    prior = prior / sum(prior)
    prevalence = module.Concatenator(input=metanet, components={"in_column": "embeddings", "out_column": "embeddings", "concatenate_column": "prevalence"})
    posterior = module.PosteriorNet(input=prevalence, components={'widths': [1055, 100]})    
    classifier = module.DiseaseClassifier(input=posterior, components={'in_width': 100, 'out_width': 55})
    data = get_test_examples()
    batch = data[0:10]
    batch['prevalence'] = prior.expand(len(batch), len(prior))
    if torch.cuda.is_available():
        metanet.cuda()
        prevalence.cuda()
        posterior.cuda()
        classifier.cuda()
        batch.cuda()
    
    output = classifier(batch)