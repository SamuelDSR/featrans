from sparsevector import SparseVector
from base import Transformer
from operator import itemgetter


class MapIndexer(Transformer):
    def __init__(self):
        pass

    def _transform(self, feature):
        if feature is None:
            return SparseVector(len(self.map_dict), [], [])
        else:
            sorted_kvs = []
            for key, val in feature.iteritems():
                sorted_kvs.append((self.map_dict.get(key,
                    self.map_dict[self.unknown_key]), val))
            sorted_kvs.sort(key = itemgetter(0))
            return SparseVector(len(self.map_dict), 
                map(itemgetter(0), sorted_kvs),
                map(itemgetter(1), sorted_kvs))
