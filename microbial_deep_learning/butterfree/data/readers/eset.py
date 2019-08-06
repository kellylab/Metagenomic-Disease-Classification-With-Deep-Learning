
class eset:
    """ Represents eset objects from curatedMetagenomicData in R. """
    
    def __init__(self, experiment_data = None, assay_data = None, pheno_data = )

        self.slot_keys = ('experiment_data', 'assay_data', 'pheno_data',
         'feature_data', 'annotation', 'protocol_data', '__class_version')
        self.slot_types = {'experiment_data' : MIAME, 'assay_data' : matrix, 'pheno_data' : ,
         'feature_data' : , 'annotation' : , 'protocol_data' : , '__class_version'}