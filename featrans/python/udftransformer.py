from base import Transformer


class UDFTransformer(Transformer):
    def __init__(self):
        pass

    def _transform(self, **kwargs):
        return self.func(**kwargs)

    def transform(self, feature_dict):
        for inputCol in self.inputCol_list:
            if inputCol not in feature_dict:
                raise KeyError("Raw feature %s was not found in sample %s"
                        % (inputCol, feature_dict))
        call_args = {}
        for f_arg, feature_name in self.args_map_dict:
            call_args[f_arg] = feature_dict[feature_name]
        feature_dict[self.outputCol]= self._transform(**call_args)
        return feature_dict
