from pyspark import StorageLevel
from base import SparkTransformer
import dill

class SparkPipeline(object):
    def __init__(self, stages = None):
        self._stages = stages

    def addStage(self, stage):
        if self._stages is None:
            self._stages = []
        self._stages.append(stage)

    def transform(self, dataset):
        for stage in self._stages:
            print("Transform: (stage: %s), (feature output: %s)" % (stage.name, stage.outputCol))
            if isinstance(stage, SparkTransformer):
                dataset = stage.transform(dataset)
            else:
                pass
        return dataset
    
    def saveModel(self, path):
        pipeline_model = []
        for stage in self._stages:
            pipeline_model.append(stage.save_as_dict())
        with open(path, 'w') as writer:
            dill.dump(pipeline_model, writer)
