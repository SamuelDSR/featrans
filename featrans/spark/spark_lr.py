from pyspark.ml.classification import LogisticRegression

class SparkLogisticRegressionWrapper(object):
    def __init__(self, featuresCol = 'feature', labelCol = 'label',\
            maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-6,\
            rawPredictionCol = 'rawPredictionCol', probabilityCol = 'probability',\
            predictionCol = 'prediction'):
        self.name = 'LogisticRegression'
        self.featuresCol = featuresCol
        self.labelCol = labelCol
        self.maxIter = maxIter
        self.regParam = regParam
        self.elasticNetParam = elasticNetParam
        self.tol = tol
        self.rawPredictionCol = rawPredictionCol
        self.probabilityCol = probabilityCol
        self.predictionCol = predictionCol
        self.lr_estimator = LogisticRegression(featuresCol = featuresCol, labelCol = labelCol,\
                maxIter = maxIter,regParam = regParam,elasticNetParam = elasticNetParam,tol = tol,\
                rawPredictionCol = rawPredictionCol, probabilityCol = probabilityCol, predictionCol = probabilityCol)

    def fit(self, dataset):
        self.lr_model = self.lr_estimator.fit(dataset)
        return self.lr_model

    def transform(self, dataset):
        if self.lr_model is None:
            raise Exception("The logistic regression model was not fitted yet, and therefore None")
        else:
            dataset = self.lr_model.transform(dataset)
            return dataset

    def saveAsDict(self):
        model = {}
        model['name'] = self.name
        model['featuresCol'] = self.featuresCol
        model['labelCol'] = self.labelCol
        model['rawPredictionCol'] = self.rawPredictionCol
        model['probabilityCol'] = self.probabilityCol
        model['predictionCol'] = self.predictionCol
        model['regParam'] = self.regParam
        model['elasticNetParam'] = self.elasticNetParam
        model['tol'] = self.tol
        if self.nb_model is not None:
            model['coefficients'] = self.lr_estimator.coefficients.toArray().tolist()
            model['intercept'] = self.intercept
        return model
