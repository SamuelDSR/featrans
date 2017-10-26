from onehot import OneHotEncoder
from combiner import Combiner
from bucketizer import Bucketizer
from vectorassembler import VectorAssembler
from udftransformer import UDFTransformer
from typeconverter import TypeConverter 
from objectindexer import ObjectIndexer
from listindexer import ListIndexer
from mapindexer import MapIndexer
from sparsevector import SparseVector

import dill
import sys


class Pipeline(object):

    def __init__(self, stages = None):
        self.stages = stages

    def addStage(self, stage):
        if self.stages is None:
            self.stages = []
        self.stages.append(stage)

    def get_model(self, model_dict):
        #model_class = getattr(importlib.import_module(".", __name__), model_dict['name'])
        model_class = eval(model_dict['name'])
        model = model_class()
        model.load_from_dict(model_dict)
        return model
        
    def load(self, path):
        with open(path, 'r') as f:
            model_list = dill.load(f)
            if self.stages is None:
                self.stages = []
            for model_dict in model_list:
                self.stages.append(self.get_model(model_dict))

    def save(self, path):
        with open(path, 'w') as f:
            model_list = []
            for stage in self.stages:
                model_list.append(stage.save_to_dict())
            dill.dump(model_list, f)

    def transform(self, feature_dict, check_equality=False):
        for stage in self.stages:
            if not check_equality:
                feature_dict = stage.transform(feature_dict)
            else:
                original = feature_dict[stage.outputCol]
                feature_dict = stage.transform(feature_dict)
                reconstruc = feature_dict[stage.outputCol]
                if original != reconstruc:
                    print("Stage info: (name: %s, outputCol: %s)" % (stage.name, stage.outputCol))
                    print("Original: %s" % original)
                    print("Reconstruct: %s" % reconstruc)
                    sys.exit(0)
        return feature_dict
