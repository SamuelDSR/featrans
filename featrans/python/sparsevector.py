import json
from operator import itemgetter

class SparseVector(object):

    def __init__(self, size = 0, indices = [], values = []):
        self.size = size
        self.ivmap = {}
        if len(indices) != len(values) or size < len(indices):
            raise Exception("SparseVector init: size doesn't match")
        else:
            for i, indice in enumerate(indices):
                if values[i] != 0:
                    self.ivmap[indice] = values[i]

    def __getitem__(self, ind):
        if ind > self.size-1:
            raise Exception("Index out of boundary")
        if ind in self.ivmap:
            return self.ivmap[ind]
        else:
            return 0.0

    def __setitem__(self, ind, val):
        self.ivmap[ind] = val

    def __repr__(self):
        json_dict = {}
        json_dict['size'] = self.size
        json_dict['type'] = 0
        ind_val_list = self.ivmap.items()
        ind_val_list.sort(key=itemgetter(0))
        json_dict['indices'] = list(map(itemgetter(0), ind_val_list))
        json_dict['values'] = list(map(itemgetter(1), ind_val_list))
        return json.dumps(json_dict)

    def __eq__(self, other):
        if self.size != other.size or len(self.ivmap) != len(other.ivmap):
            return False
        for key in self.ivmap:
            if key not in other.ivmap or self.ivmap[key] != other.ivmap[key]:
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def __le__(self, other):
        return self.size <= other.size

    def set_ivmap(self, ivmap):
        self.ivmap = ivmap

    def insert(self, idx, val):
        if idx < self.size and val != 0:
            self.ivmap[idx] = val

    @classmethod
    def empty_vector(cls):
        return SparseVector(0, [], [])

    def concat(self, sparse_vec):
        for indice in sparse_vec.ivmap:
            self.ivmap[indice + self.size] = sparse_vec.ivmap[indice]
        self.size += sparse_vec.size
        return self

    def dot(self, sparse_vec):
        if self.size != sparse_vec.size:
            raise Exception("Size of Sparsevector doesn't match for dot product")
        else:
            dot_sum = 0
            for indice in self.ivmap:
                if indice in sparse_vec.ivmap:
                    dot_sum += self.ivmap[indice]*sparse_vec.ivmap[indice]
            return dot_sum
    
    def to_dense(self):
        try:
            import numpy as np
            dense_vec = np.zeros(self.size, 1)
            for indice, value in self.ivmap.iteritems():
                dense_vec[indice] = value
            return dense_vec
        except:
            dense_vec = [0.0]*self.size
            for indice, value in self.ivmap.iteritems():
                dense_vec[indice] = value
            return dense_vec

class SparseVectorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SparseVector):
            return obj.__repr__()
        return super(SparseVectorEncoder, self).default(obj)

class SparseVectorDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "type" in obj and obj["type"] == 0 and "size" in obj and "indices" in obj and "values" in obj:
            return SparseVector(obj['size'], obj['indices'], obj['values'])
        else:
            return obj
