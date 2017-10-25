import itertools
from exceptions import KeyError
from sparsevector import SparseVector

import math

class NaiveBayes(object):
    def __init__(self, featuresCol = 'feature', labelCol = 'label',\
            rawPredictionCol = 'rawPrediction', probabilityCol = 'probability',\
            predictionCol = 'prediction', smoothing = 1.0, modelType = 'multinomial', \
            pi = None, theta = None):
        self.featuresCol = featuresCol
        self.labelCol = labelCol
        self.rawPredictionCol = rawPredictionCol
        self.predictionCol = predictionCol
        self.probabilityCol = probabilityCol
        self.smoothing = smoothing
        self.modelType = modelType
        self.pi = pi
        self.theta = theta


    def loadFromDict(self, model_dict):
        if cmp(self.__class__.__name__, model_dict['name']) == 0:
            self.featuresCol = model_dict['featuresCol']
            self.labelCol = model_dict['labelCol']
            self.rawPredictionCol = model_dict['rawPredictionCol']
            self.probabilityCol = model_dict['probabilityCol']
            self.predictionCol = model_dict['predictionCol']
            self.smoothing = model_dict['smoothing']
            self.modelType = model_dict['modelType']
            self.pi = model_dict['pi']
            self.theta = model_dict['theta']


    def saveAsDict(self):
        ret = {}
        ret['featuresCol']  = self.featuresCol
        ret['labelCol'] = self.labelCol
        ret['smoothing'] = self.smoothing
        ret['modelType'] = self.modelType
        ret['rawPredictionCol'] = self.rawPredictionCol
        ret['predictionCol'] = self.predictionCol
        ret['probabilityCol'] = self.probabilityCol
        ret['pi'] = self.pi
        ret['theta'] = self.theta
        return ret

    def _transform(self, input_feature):
        rawPrediction = []
        for class_prior, feature_prior in itertools.izip(self.pi, self.theta):
            if isinstance(input_feature, SparseVector):
                if input_feature.size != len(feature_prior):
                    raise Exception("Serious error, model size and feature size doesn't match")
                prob = class_prior
                for idx in input_feature.ivmap:
                    prob += feature_prior[int(idx)]
            else:
                prob = class_prior
                for a,b in itertools.izip(input_feature, feature_prior):
                    prob += a*b
            rawPrediction.append(prob)
        ratio = math.exp(rawPrediction[0] - rawPrediction[1])
        probability = [ratio/(ratio + 1), 1/(ratio + 1)]
        return (rawPrediction, probability)

    def transform(self, feature_dict):
        if self.featuresCol not in feature_dict:
            raise KeyError("features col %s was not found" % self.featuresCol)
        else:
            rawPrediction, probability = self._transform(feature_dict[self.featuresCol])
            feature_dict[self.rawPredictionCol] = rawPrediction
            feature_dict[self.probabilityCol] = probability
            return feature_dict
