from base import SparkTransformer

class SparkTypeConverter(SparkTransformer):
    def __init__(self, inputCol=None, outputCol=None, outputType = None):
        self.name = 'TypeConverter'
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.outputType = outputType

    def transform(self, dataset):
        dataset = dataset.withColumn(self.inputCol, dataset[self.outputCol].cast(self.outputType))
        return dataset

    def save_as_dict(self):
        ret = super(SparkTypeConverter, self).save_as_dict()
        ret['outputType'] = self.outputType.__class__.__name__
        return ret