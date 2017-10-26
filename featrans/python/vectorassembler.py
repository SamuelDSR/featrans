from exceptions import KeyError
from collections import Iterable

from sparsevector import SparseVector
from base import Transformer

class VectorAssembler(Transformer):
    """
    VectorAssembler accept: SparseVector, single element (int, float), and enumerable (list, tuple)
    1. all basic elements in SparseVector or list(tuple) are automatically converted to float
    """

    def __init__(self):
        pass

    def _transform(self, vec_list):
        if len(vec_list) == 0:
            return SparseVector.empty_vector()

        final_sparse_vec = SparseVector.empty_vector()
        for vec in vec_list:
            #SparseVector
            if isinstance(vec, SparseVector):
                final_sparse_vec.concat(vec)
            #Iterable
            elif isinstance(vec, Iterable):
                tmp_sparse_vec = SparseVector(len(vec), 
                        range(0, len(vec)), map(lambda x: float(x), vec))
                final_sparse_vec.concat(tmp_sparse_vec)
            #single element
            else:
                tmp_sparse_vec = SparseVector(1, [0], [float(vec)])
                final_sparse_vec.concat(tmp_sparse_vec)
        return final_sparse_vec


    def transform(self, feature_dict):
        vec_list = []
        for inputCol in self.inputCol_list:
            if inputCol not in feature_dict:
                raise KeyError("Raw feature %s was not found in sample %s"
                        % (inputCol, feature_dict))
            else:
                vec_list.append(feature_dict[inputCol])
        feature_dict[self.outputCol] = self._transform(vec_list)
        return feature_dict
