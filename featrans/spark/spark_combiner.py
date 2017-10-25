from pyspark.sql.types import StringType
from pyspark.sql.functions import udf, col
from base import SparkTransformer


class SparkCombiner(SparkTransformer):

    def __init__(self, inputCol_list=None, outputCol=None):
        self.inputCol_list = inputCol_list
        self.outputCol = outputCol
        self.name = 'Combiner'

    def transform(self, dataset):
        combiner_udf = udf(lambda x, y: str(x)+"-"+str(y), StringType())
        return dataset.withColumn(self.outputCol, combiner_udf(col(self.inputCol_list[0]),\
                col(self.inputCol_list[1])))
