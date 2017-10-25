class SparseVector(object):

    def __init__(self, size = 0, indices = [], values = []):
        self.size = size
        self.ivmap = {}
        if len(indices) != len(values) or size < len(indices):
            raise Exception("SparseVector init: size doesn't match")
        else:
            for i, indice in enumerate(indices):
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
        import json
        return json.dumps(self.__dict__)

    def set_ivmap(self, ivmap):
        self.ivmap = ivmap

    def insert(self, idx, val):
        self.ivmap[idx] = val

    @classmethod
    def from_json(obj_str):
        import json
        obj_dict = json.loads(obj_str)
        retvec = SparseVector(size = obj_dict['size'])
        retvec.setIvmap(obj_dict['ivmap'])
        return retvec

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
