from base import Transformer
from sparsevector import SparseVector


class ListIndexer(Transformer):
    def __init__(self):
        pass

    def _transform(self, feature):
        if feature is None:
            return SparseVector(len(self.map_dict), [], [])
        else:
            feature = map(lambda x: self.map_dict.get(x,
                self.map_dict[self.unknown_key]), feature)
            feature = sorted(feature)
            return SparseVector(len(self.map_dict), feature, [1.0]*len(feature))
