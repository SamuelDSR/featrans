from base import SparkTransformer

class SparkTypeConverter(SparkTransformer):
    def __init__(self, inputCol=None, outputCol=None, outputType = None):
        self.name = 'TypeConverter'
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.outputType = outputType

    def transform(self, dataset):
        _t_inputCol = self.inputCol
        _t_outputCol = self.outputCol
        _t_outputType = self.outputType

        dataset = dataset.withColumn(_t_outputCol, dataset[_t_inputCol].cast(_t_outputType))
        return dataset

    def save_as_dict(self):
        ret = {}
        ret['name'] = self.name
        ret['inputCol'] = self.inputCol
        ret['outputCol'] = self.outputCol
        ret['outputType'] = self.outputType.__class__.__name__
        return ret
