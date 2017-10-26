from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf, col

from base import SparkTransformer
from functools import partial

class SparkListIndexer(SparkTransformer):
    def __init__(self, inputCol=None, outputCol=None, map_dict=None, unknown_key=None):
        self.name = "ListIndexer"
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.map_dict = map_dict
        self.unknown_key = unknown_key

    def transform(self, dataset):
        def list_type_map_func(key_list, map_dict, unknown_key):
            keys = []
            for key in key_list:
                keys.append(map_dict.get(key, map_dict[unknown_key]))
            #may contain duplicated key in key_list
            keys = list(set(keys))
            keys.sort()
            return Vectors.sparse(len(map_dict), keys, [1.0]*len(keys))

        if self.map_dict is None:
            all_keys = dataset.filter(col(self.inputCol).isNotNull())\
                    .select(self.inputCol).rdd\
                    .flatMap(lambda x: x[self.inputCol])\
                    .distinct().collect()
            all_keys.append(self.unknown_key)
            self.map_dict = dict((tup[1], tup[0]) for tup in enumerate(all_keys))

        map_udf = udf(partial(list_type_map_func,
            unknown_key=self.unknown_key, map_dict=self.map_dict), VectorUDT())
        return dataset.withColumn(self.outputCol, map_udf(col(self.inputCol)))
