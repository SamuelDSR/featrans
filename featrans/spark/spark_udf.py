from pyspark.sql.types import *
from pyspark.sql.functions import udf, struct

import inspect
from functools import partial
from base import SparkTransformer

class SparkUDFTransformer(SparkTransformer):

    def __init__(self, inputCol_list = None, outputCol = None,\
            func = None,  outputType = None, args_map_dict = None,):
        self.name = 'UDFTransformer'
        self.func = func
        self.inputCol_list = inputCol_list
        self.outputCol = outputCol
        self.outputType = outputType
        self.args_map_dict = args_map_dict
        if self.args_map_dict is None:
            func_args = inspect.getargspec(self.func).args
            self.args_map_dict = dict(zip(func_args, self.inputCol_list))

    def transform(self, dataset):
        _t_inputCol_list = self.inputCol_list
        _t_outputCol = self.outputCol
        _t_outputType = self.outputType
        _t_args_map_dict = self.args_map_dict
        _t_func = self.func

        def map_func(key_val_dict, func, args_map_dict):
            key_val_dict = key_val_dict.asDict()
            call_args = {}
            for f_arg, col_name in args_map_dict.iteritems():
                call_args[f_arg] = key_val_dict[col_name]
            return func(**call_args)

        map_udf = udf(partial(map_func, func=_t_func, 
            args_map_dict=_t_args_map_dict), _t_outputType)
        return dataset.withColumn(_t_outputCol, map_udf(struct(_t_inputCol_list)))

    def save_as_dict(self):
        ret = {}
        ret['name'] = self.name
        ret['func'] = self.func
        ret['inputCol_list'] = self.inputCol_list
        ret['outputCol'] = self.outputCol
        ret['args_map_dict'] = self.args_map_dict
        ret['outputType'] = self.outputType.__class__.__name__
        return ret
