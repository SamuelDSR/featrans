from pyspark.ml.linalg import VectorUDT
from pyspark.sql.functions import udf, col

from functools import partial
from base import SparkTransformer

class SparkObjectIndexer(SparkTransformer):
    def __init__(self, inputCol=None, outputCol=None, map_dict=None, unknown_key=None):
        self.name = 'ObjectIndexer'
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.map_dict = map_dict
        self.unknown_key = unknown_key

    def transform(self, dataset):
        def map_func(key, map_dict, unknown_key):
            return map_dict.get(key, map_dict[unknown_key])

        if self.map_dict is None:
            all_keys = dataset.filter(col(self.inputCol).isNotNull())\
                    .select(self.inputCol).rdd\
                    .map(lambda x: x[self.inputCol])\
                    .distinct().collect()
            all_keys.append(self.unknown_key)
            self.map_dict = dict((tup[1], tup[0]) for tup in enumerate(all_keys))

        map_udf = udf(partial(map_func, map_dict=self.map_dict, 
            unknown_key=self.unknown_key), VectorUDT())
        return dataset.withColumn(self.outputCol, map_udf(col(self.inputCol)))
