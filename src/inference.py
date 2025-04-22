from paths import model_path
import joblib
import pandas as pd

model = joblib.load(model_path)

feature_names = [
    "Age",
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
]


def GPAGradeClass(GPA: float) -> int:
    if GPA < 2:
        return 4
    elif GPA < 2.5:
        return 3
    elif GPA < 3:
        return 2
    elif GPA < 3.5:
        return 1
    else:
        return 0


def GradeClassLabel(GradeClass: int) -> str:
    return ['A', 'B', 'C', 'D', 'F'][GradeClass]


def FeatureEngineer(x: pd.DataFrame) -> pd.DataFrame:
    ret = x.copy(deep=True)
    ret['TotalExtracurricular'] = ret[['Extracurricular', 'Sports', 'Music', 'Volunteering']].sum(axis=1)
    ret['Tutoring_ParentalSupport'] = ret['Tutoring'] * ret['ParentalSupport']

    return ret


def Predict(x: pd.DataFrame) -> pd.DataFrame:
    return model.predict(x)