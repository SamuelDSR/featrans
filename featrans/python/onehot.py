from base import Transformer
from sparsevector import SparseVector

class OneHotEncoder(Transformer):
    def __init__(self):
        pass

    def _transform(self, feature):
        return SparseVector(self.size, [feature], [1.0])
