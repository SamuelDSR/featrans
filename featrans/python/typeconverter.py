from exceptions import TypeError
from base import Transformer

type_mapping_table = {'DoubleType': 'float', 'FloatType': 'float', 'IntegerType': 'int',\
        'StringType' : 'str', 'LongType': 'long', 'ArrayType': 'list'}

class TypeConverter(Transformer):
    def __init__(self):
        pass

    def _transform(self, feature_val):
        try:
            return eval(type_mapping_table[self.outputType])(feature_val)
        except:
            raise TypeError("type converter failed for type %s and value %s"\
                    % (self.outputCol, feature_val))
