import os

script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(script_dir, "..", "artifacts", "random_forest_regressor.joblib")
data_path = os.path.join(script_dir, "..", "artifacts", "Student_performance_data.csv")
