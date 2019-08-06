import numpy as np
import pandas as pd
import os
from Reader import Reader
from eset import eset

class esetReader(Reader):
    def __init__(self, eset = None):
        Reader.__init__(self)
        if eset = None:
            self.eset = eset()
        else:
            self.eset = eset
        self.MIAME = MIAME()
        self.MIAMEReader = MIAMEReader(self.MIAME)

    def read_file(self, file):
        if file in self.eset.slot_keys:
            if self.eset.slot_types[file] == MIAME:
                self.MIAMEReader.read_file(file)
                self.eset.file = self.MIAME
            elif: self.eset.slot_types[file] ==
