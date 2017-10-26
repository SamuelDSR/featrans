from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf, col

from functools import partial
from base import SparkTransformer

class SparkOneHotEncoder(SparkTransformer):
    def __init__(self, inputCol=None, outputCol=None, size = None):
        self.name = 'OneHotEncoder'
        self.size = size
        self.inputCol = inputCol
        self.outputCol = outputCol

    def transform(self, dataset):
        _t_inputCol = self.inputCol
        _t_outputCol = self.outputCol
        _t_size = self.size

        def onehot_map_func(val, size):
            return Vectors.sparse(size, [val], [1.0])

        if self.size is None:
            self.size = dataset.select(_t_inputCol).rdd\
                    .map(lambda x: x[_t_inputCol]).max()
        map_udf = udf(partial(onehot_map_func, size=_t_size), VectorUDT())
        return dataset.withColumn(_t_outputCol, map_udf(col(_t_inputCol)))
