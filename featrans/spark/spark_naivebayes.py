from pyspark.ml.classification import NaiveBayes

class SparkNaiveBayesWrapper(object):
    def __init__(self, featuresCol = 'feature', labelCol = 'label',\
            rawPredictionCol = 'rawPrediction', probabilityCol = 'probability',\
            predictionCol = 'prediction', smoothing = 1.0, modelType = 'multinomial'):
        self.name = 'NaiveBayes'
        self.featuresCol = featuresCol
        self.labelCol = labelCol
        self.rawPredictionCol = rawPredictionCol
        self.probabilityCol = probabilityCol
        self.predictionCol = predictionCol
        self.smoothing = smoothing
        self.modelType = modelType
        self.nb_estimator = NaiveBayes(featuresCol = featuresCol, labelCol = labelCol,\
                smoothing = smoothing, modelType = modelType, rawPredictionCol = rawPredictionCol,\
                probabilityCol = probabilityCol, predictionCol = predictionCol)
        self.nb_model = None

    def fit(self, dataset):
        self.nb_model = self.nb_estimator.fit(dataset)
        return self.nb_model


    def transform(self, dataset):
        if self.nb_model is None:
            raise Exception("The naive bayes model was not fitted yet, and therefore None")
        else:
            dataset = self.nb_model.transform(dataset)
            return dataset

    def saveAsDict(self):
        model = {}
        model['name'] = self.name
        model['featuresCol'] = self.featuresCol
        model['labelCol'] = self.labelCol
        model['smoothing'] = self.smoothing
        model['modelType'] = self.modelType
        model['rawPredictionCol'] = self.rawPredictionCol
        model['probabilityCol'] = self.probabilityCol
        model['predictionCol'] = self.predictionCol
        if self.nb_model is not None:
            model['pi'] = self.nb_model.pi.toArray().tolist()
            model['theta'] = self.nb_model.theta.toArray().tolist()
        return model
