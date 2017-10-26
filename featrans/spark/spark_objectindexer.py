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
        _t_unknown_key = self.unknown_key
        _t_map_dict = self.map_dict
        _t_inputCol = self.inputCol
        _t_outputCol = self.outputCol

        def map_func(key, map_dict, unknown_key):
            return map_dict.get(key, map_dict[unknown_key])

        if self.map_dict is None:
            all_keys = dataset.filter(col(_t_inputCol).isNotNull())\
                    .select(_t_inputCol).rdd\
                    .map(lambda x: x[_t_inputCol])\
                    .distinct().collect()
            all_keys.append(_t_unknown_key)
            self.map_dict = dict((tup[1], tup[0]) for tup in enumerate(all_keys))


        map_udf = udf(partial(map_func, map_dict=_t_map_dict, 
            unknown_key=_t_unknown_key), VectorUDT())
        return dataset.withColumn(_t_outputCol, map_udf(col(_t_inputCol)))
