from Reader import Reader
from experiment_hub import experiment_hub
from eset import eset
from esetReader import esetReader

class experimentHubReader(Reader):

    def __init__(self, experiment_hub = None):
        Reader.__init__(self)
        if experiment_hub = None:
            self.experiment_hub = experiment_hub()
        else:
            self.experiment_hub = experiment_hub
        self.eset = eset()
        self.esetReader = esetReader(self.eset)

    def read_file(file):
        """ Determines what type of object the file represents and calls the appropriate reader. """
        if file in self.experiment_hub.keys:
            csv = self.read_csv()
            self.experiment_hub.file = csv
        else: # Is an eset file
            self.esetReader.read_file(file)

    def read_eset(self, eset):
        """ Reads eset object and loads it into the experiment_hub as an attribute. """
        pass

def read_csv(csv, delimiter = ','):
    """ Reads csv files that were written in R and returns ndarrays. """
    if type(csv) == str:
        csvfile = open(csv)
    else:
        csvfile = csvfile # Assume csv is an opened file
    reader = csv.reader(csvfile, delimiter = delimiter)
    reading = []
    for row in reader:
        reading.append(row)

    return reading

def read_eset(eset):
    """ Reader for eset objects. Only loads annotated dataframe attributes and not slots. """
    pass

def read_MIAME(MIAME):
    """ Reader for MIAME objects. Not implemented yet. """
    pass

def read_data_frame(data_frame):
    """ Reads (annotated) data_frames written in R and returns pandas dataframes """
    pass
