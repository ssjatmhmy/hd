# Predict with different estimators
ypreds = {}
for est in [XGBEstimator(), LassoEstimator(), RidgeEstimator(), RFREstimator()]:
    model = est.train(nd_train, nd_label)
    ypred = est.predict(model, nd_test)
    ypreds[est.name] = ypred
