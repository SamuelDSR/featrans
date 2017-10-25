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
        def onehot_map_func(val, size):
            return Vectors.sparse(size, [val], [1.0])

        if self.size is None:
            self.size = dataset.select(self.inputCol).rdd\
                    .map(lambda x: x[self.inputCol]).max()
        map_udf = udf(partial(onehot_map_func, size=self.size), VectorUDT())
        return dataset.withColumn(self.outputCol, map_udf(col(self.inputCol)))
