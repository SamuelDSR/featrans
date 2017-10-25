from base import Transformer

class ObjectIndexer(Transformer):
    def __init__(self):
        pass

    def _transform(self, feature):
        return self.map_dict.get(feature, self.map_dict[self.unknown_key])
