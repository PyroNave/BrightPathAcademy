import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score

class CustomModel(BaseEstimator, ClassifierMixin):
    def __init__(self, regressor, scaler):
        self.scaler = scaler
        self.regressor = regressor

    def _gpa_to_grade(self, gpa):
        if gpa < 2:
            return 4
        elif gpa < 2.5:
            return 3
        elif gpa < 3:
            return 2
        elif gpa < 3.5:
            return 1
        else:
            return 0

    def _preprocess(self, X):
        ret = X.copy(deep=True)
        ret[['StudyTimeWeekly', 'Absences']] = self.scaler.transform(ret[['StudyTimeWeekly', 'Absences']])
        ret['TotalExtracurricular'] = ret[['Extracurricular', 'Sports', 'Music', 'Volunteering']].sum(axis=1)
        ret['Tutoring_ParentalSupport'] = ret['Tutoring'] * ret['ParentalSupport']

        return ret

    def _postprocess(self, X):
        return np.array([self._gpa_to_grade(gpa) for gpa in X])

    def predict(self, X, preprocess=True):
        if preprocess:
            X = self._preprocess(X)
        gpa_pred = self.regressor.predict(X)
        gradeclass_pred = self._postprocess(gpa_pred)

        return gradeclass_pred

    def fit(self, X, y_gpa, preprocess=True):
        copy = X.copy(deep=True)

        if preprocess:
            copy[['StudyTimeWeekly', 'Absences']] = self.scaler.fit_transform(copy[['StudyTimeWeekly', 'Absences']])

        self.regressor.fit(copy, y_gpa)
        return self

    def score(self, X, gradeclass_true, preprocess=True):
        gradeclass_pred = self.predict(X, preprocess=preprocess)
        accuracy = np.mean(gradeclass_pred == gradeclass_true)
        return accuracy

    def print_report(self, X_test, y_test, preprocess=True):
        X = X_test
        if preprocess:
            X = self._preprocess(X_test)

        gpa_pred = self.regressor.predict(X)
        print("Regression metrics:")
        print(f"\tMAE: {mean_absolute_error(y_test, gpa_pred):.4f}")
        print(f"\tMSE: {mean_squared_error(y_test, gpa_pred):.4f}")
        print(f"\tRMSE: {np.sqrt(mean_squared_error(y_test, gpa_pred)):.4f}")
        print(f"\tR²: {r2_score(y_test, gpa_pred):.4f}")

        gradeclass_pred = [self._gpa_to_grade(gpa) for gpa in gpa_pred]
        gradeclass_true = [self._gpa_to_grade(gpa) for gpa in y_test]
        print("Classification Metrics:")
        print(classification_report(gradeclass_true, gradeclass_pred, target_names=['A', 'B', 'C', 'D', 'F']))
