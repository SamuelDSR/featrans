from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf, col
from base import SparkTransformer

from operator import itemgetter
from functools import partial

class SparkMapIndexer(SparkTransformer):
    def __init__(self, inputCol=None, outputCol=None, map_dict=None, unknown_key=None):
        self.name = "MapIndexer"
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.map_dict = map_dict
        self.unknown_key = unknown_key

    def transform(self, dataset):
        def map_type_map_func(val_dict, map_dict, unknown_key):
            sorted_kvs = []
            for key, val in val_dict.iteritems():
                sorted_kvs.append((map_dict.get(key, map_dict[unknown_key]), val))
            #may contain duplicated key in key_list
            sorted_kvs = list(set(sorted_kvs))
            sorted_kvs.sort(key = itemgetter(0))
            return Vectors.sparse(len(map_dict), 
                map(itemgetter(0), sorted_kvs),
                map(itemgetter(1), sorted_kvs))

        if self.map_dict is None:
            all_keys = dataset.filter(col(self.inputCol).isNotNull())\
                    .select(self.inputCol).rdd\
                    .flatMap(lambda x: x.keys())\
                    .distinct().collect()
            all_keys.append(self.unknown_key)
            self.map_dict = dict((tup[1], tup[0]) for tup in enumerate(all_keys))

        map_udf = udf(partial(map_type_map_func,
            map_dict=self.map_dict, unknown_key=self.unknown_key), VectorUDT())
        return dataset.withColumn(self.outputCol, map_udf(col(self.inputCol)))
