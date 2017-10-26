from base import SparkTransformer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf, array
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import *


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
        def map_func(val_list):
            size = 0
            indices = []
            values = []
            for val in val_list:
                if isinstance(val, IntegerType()) or isinstance(val, DoubleType())\
                        or isinstance(val, FloatType()):
                    indices.append(size)
                    values.append(float(val))
                    size += 1
                elif isinstance(val, VectorUDT()):
                    indices.extend(val.indices)
                    values.extend(map(float, val.values))
                    size += val.size
                else:
                    raise Exception("VectorAssembler only int/float/double/SparseVector")
            return Vectors.sparse(size, indices, values)
        #map_udf = udf(map_func, )
        return self.internal_transformer.transform(dataset)

    def save_as_dict(self):
        ret = {}
        ret['name'] = self.name
        ret['inputCol_list'] = self.inputCol_list
        ret['outputCol'] = self.outputCol
        return ret
