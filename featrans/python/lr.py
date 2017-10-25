import itertools
from exceptions import KeyError
from sparsevector import SparseVector

import math

class LogisticRegression(object):
    def __init__(self, featuresCol = 'feature', labelCol = 'label',\
            rawPredictionCol = 'rawPrediction', probabilityCol = 'probability',\
            coefficients = None, intercept = None):
        self.featuresCol = featuresCol
        self.labelCol = labelCol
        self.rawPredictionCol = rawPredictionCol
        self.probabilityCol = probabilityCol
        self.predictionCol = predictionCol
        self.coefficients = coefficients
        self.intercept = intercept


    def loadFromDict(self, model_dict):
        if cmp(self.__class__.__name__, model_dict['name']) == 0:
            self.featuresCol = model_dict['featuresCol']
            self.labelCol = model_dict['labelCol']
            self.rawPredictionCol = model_dict['rawPredictionCol']
            self.probabilityCol = model_dict['probabilityCol']
            self.predictionCol = model_dict['predictionCol']
            self.coefficients = model_dict['coefficients']
            self.intercept = model_dict['intercept']



    def saveAsDict(self):
        ret = {}
        ret['featuresCol'] = self.featuresCol
        ret['labelCol'] = self.labelCol
        ret['rawPredictionCol'] = self.rawPredictionCol
        ret['predictionCol'] = self.predictionCol
        ret['probabilityCol'] = self.probabilityCol
        ret['coefficients'] = self.coefficients
        ret['intercept'] = self.intercept
        return ret

    def _transform(self, input_feature):
        rawPrediction = self.intercept
        if isinstance(input_feature, SparseVector):
            if input_feature.size != len(self.coefficents):
                raise Exception("Serious error, model size and feature size doesn't match")
            for idx in input_feature.ivmap:
                rawPrediction += self.coefficents[idx]*input_feature[idx]
        else:
            for a, b in itertools.izip(input_feature, self.coefficents):
                rawPrediction += a*b
        probability = 1 / (1 + math.exp(-rawPrediction))
        return (rawPrediction, probability)

    def transform(self, feature_dict):
        if self.featuresCol not in feature_dict:
            raise KeyError("features col %s was not found" % self.featuresCol)
        else:
            rawPrediction, probability = self._transform(feature_dict[self.featuresCol])
            feature_dict[self.rawPredictionCol] = rawPrediction
            feature_dict[self.probabilityCol] = probability
            return feature_dict
