from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType
from functools import partial
from base import SparkTransformer

class SparkBucketizer(SparkTransformer):
    def __init__(self, inputCol=None, outputCol=None, splits=None):
        self.name = 'Bucketizer'
        self.splits = splits
        self.inputCol = inputCol
        self.outputCol = outputCol

    def transform(self, dataset):
        #avoid pickling the entire SparkBucketizer object in yarn mode
        _t_splits = self.splits
        _t_outputCol = self.outputCol
        _t_inputCol = self.inputCol

        def encoder_func(val, splits):
            left = 0
            right = len(splits) - 1
            while left < right:
                mid = left + (right - left)/2
                if splits[mid] < val:
                    left = mid + 1
                else:
                    right = mid
            if splits[left] > val:
                return left - 1
            else:
                return left

        map_udf = udf(partial(encoder_func, splits=_t_splits), IntegerType())
        return dataset.withColumn(_t_outputCol, map_udf(col(_t_inputCol)))
