from base import SparkTransformer
from pyspark.ml.feature import VectorAssembler


class SparkAssembler(SparkTransformer):

    """
    Vector Assembler only supports:
    1. single element (IntegerType(), FloatType(), etc), and will automatically convert it to float
    2. SparseVector
    """

    def __init__(self, inputCol_list = None, outputCol = None):
        self.name = "VectorAssembler"
        self.inputCol_list = inputCol_list
        self.outputCol = outputCol
        self.internal_transformer = VectorAssembler(inputCols = inputCol_list, outputCol = outputCol)

    def transform(self, dataset):
        return self.internal_transformer.transform(dataset)

    def save_as_dict(self):
        ret = {}
        ret['name'] = self.name
        ret['inputCol_list'] = self.inputCol_list
        ret['outputCol'] = self.outputCol
        return ret
