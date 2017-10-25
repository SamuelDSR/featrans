from exceptions import KeyError


class Transformer(object):
    def __init__(self):
        pass

    def load_from_dict(self, attr_dict):
        self.__dict__.update(attr_dict)

    def save_to_dict(self):
        return self.__dict__

    def transform(self, feature_dict):
        if self.inputCol not in feature_dict:
            raise KeyError("Raw feature %s was not found in sample %s"
                    % (self.inputCol, feature_dict))
        else:
            feature_dict[self.outputCol] = self._transform(feature_dict[self.inputCol])
        return feature_dict
