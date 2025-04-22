import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class CustomModel(BaseEstimator, ClassifierMixin):
    def __init__(self, regressor, scaler=None):
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
        ret = X.copy(deep=True)[self.feature_names_in_]
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

        return pd.DataFrame({'GPA': gpa_pred, 'GradeClass': gradeclass_pred})

    def fit(self, X, y_gpa, preprocess=True):
        copy = X.copy(deep=True)
        if not preprocess:
            self.feature_names_in_ = copy.drop(['TotalExtracurricular', 'Tutoring_ParentalSupport'], axis=1).columns
        else:
            self.feature_names_in_ = copy.columns

        if self.scaler == None:
            scaler = StandardScaler()
            scaler.fit(copy[['StudyTimeWeekly', 'Absences']])
            self.scaler = scaler

        if preprocess:
            copy = self._preprocess(copy)

        self.regressor.fit(copy, y_gpa)
        return self

    def score(self, X, gradeclass_true, preprocess=True):
        gradeclass_pred = self.predict(X, preprocess=preprocess)['GradeClass']
        accuracy = np.mean(gradeclass_pred == gradeclass_true)
        return accuracy

    def print_report(self, X_test, y_test, preprocess=True):
        y_pred = self.predict(X_test, preprocess=preprocess)
        gpa_pred = y_pred['GPA']
        gradeclass_pred = y_pred['GradeClass']

        print("Regression metrics:")
        print(f"\tMAE: {mean_absolute_error(y_test, gpa_pred):.4f}")
        print(f"\tMSE: {mean_squared_error(y_test, gpa_pred):.4f}")
        print(f"\tRMSE: {np.sqrt(mean_squared_error(y_test, gpa_pred)):.4f}")
        print(f"\tR²: {r2_score(y_test, gpa_pred):.4f}")

        gradeclass_true = [self._gpa_to_grade(gpa) for gpa in y_test]
        print("Classification Metrics:")
        print(classification_report(gradeclass_true, gradeclass_pred, target_names=['A', 'B', 'C', 'D', 'F']))
