import os
import baba.config

class Reader:
    def __init__(self, directory= os.getcwd()):
        self.directory = directory
        
    def read_directory(self, directory = None):
        """ Traverses directory and reads and loads into experiment_hub object the various parameters """
        if directory is None:
            directory = self.directory

        self.listdir = os.listdir()
        for f in self.listdir:
            full_path = os.path.join(directory, f)
            if os.path.isfile_(full_path):
                self.read_file(full_path)
            else:
                self.read_directory(full_path) # Keep iterating through the file tree

    def read_file(self, file):
        raise NotImplementedError("Must implement a method for reading files.")

    def read_csv(self, csv):
        """ Reads csv files that were written in R and returns ndarrays"""
        if type(csv) == str:
            csvfile = open(csv)
        else:
            csvfile = csvfile # Assume csv is an opened file
        reader = csv.reader(csvfile)
        reading = []
        for row in reader:
            reading.append(row)

        return reading
