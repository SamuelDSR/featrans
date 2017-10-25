from base import Transformer
from sparsevector import SparseVector


class ListIndexer(Transformer):
    def __init__(self):
        pass

    def _transform(self, feature):
        if feature is None:
            return SparseVector(len(self.map_dict), [], [])
        else:
            feature = sorted(feature)
            return SparseVector(len(self.map_dict), feature, [1]*len(self.map_dict))
