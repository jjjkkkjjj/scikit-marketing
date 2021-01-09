from sklearn.base import BaseEstimator, RegressorMixin

class ModelingMixin(RegressorMixin):
    """Mixin class for all regression estimators in scikit-learn."""
    _estimator_type = "modeler"

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import mean_squared_error
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred, sample_weight=sample_weight)