from base import Transformer
from exceptions import KeyError


class Combiner(Transformer):
    def __init__(self):
        pass

    def _transform(self, feature_list):
        return str(feature_list[0]) + "-" + str(feature_list[1])

    def transform(self, feature_dict):
        for inputCol in self.inputCol_list:
            if inputCol not in feature_dict:
                raise KeyError("Raw feature %s was not found in sample %s"
                        % (inputCol, feature_dict))
        else:
            feature_list = map(lambda x: feature_dict[x], self.inputCol_list)
            feature_dict[self.outputCol] = self._transform(feature_list)
