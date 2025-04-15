import dash
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
from dash import html, dcc
from dash.dependencies import Input, Output, State
import os

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(script_dir, "..", "artifacts", "random_forest_regressor.joblib")

model = joblib.load(model_path)

INPUT_FIELDS = [
    {
        "label": "Age",
        "id": "age",
        "type": "number",
        "default": 18,
        "component": dcc.Input,
        "props": {"type": "number", "min": 0},
    },
    {
        "label": "Gender",
        "id": "gender",
        "type": "dropdown",
        "default": 0,
        "component": dcc.Dropdown,
        "props": {
            "options": [
                {"label": "Male", "value": 0},
                {"label": "Female", "value": 1},
            ],
        },
    },
    {
        "label": "Ethnicity",
        "id": "ethnicity",
        "type": "dropdown",
        "default": 0,
        "component": dcc.Dropdown,
        "props": {
            "options": [
                {"label": "Caucasian", "value": 0},
                {"label": "African American", "value": 1},
                {"label": "Asian", "value": 2},
                {"label": "Other", "value": 3},
            ],
        },
    },
    {
        "label": "Parental Education",
        "id": "parent-edu",
        "type": "dropdown",
        "default": 0,
        "component": dcc.Dropdown,
        "props": {
            "options": [
                {"label": "None", "value": 0},
                {"label": "High School", "value": 1},
                {"label": "Some College", "value": 2},
                {"label": "Bachelor's", "value": 3},
                {"label": "Higher Study", "value": 4},
            ],
        },
    },
    {
        "label": "Study Time Weekly (hrs)",
        "id": "study-time",
        "type": "number",
        "default": 0,
        "component": dcc.Input,
        "props": {"type": "number", "min": 0},
    },
    {
        "label": "Absences",
        "id": "absences",
        "type": "number",
        "default": 0,
        "component": dcc.Input,
        "props": {"type": "number", "min": 0},
    },
    {
        "label": "Tutoring",
        "id": "tutoring",
        "type": "radio",
        "default": 0,
        "component": dcc.RadioItems,
        "props": {
            "options": [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
            "inline": True,
        },
    },
    {
        "label": "Parental Support",
        "id": "parent-support",
        "type": "radio",
        "default": 0,
        "component": dcc.RadioItems,
        "props": {
            "options": [
                {"label": "None", "value": 0},
                {"label": "Low", "value": 1},
                {"label": "Moderate", "value": 2},
                {"label": "High", "value": 3},
                {"label": "Very High", "value": 4},
            ],
        },
    },
    {
        "label": "Extracurricular",
        "id": "extracurricular",
        "type": "radio",
        "default": 0,
        "component": dcc.RadioItems,
        "props": {
            "options": [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
            "inline": True,
        },
    },
    {
        "label": "Sports",
        "id": "sports",
        "type": "radio",
        "default": 0,
        "component": dcc.RadioItems,
        "props": {
            "options": [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
            "inline": True,
        },
    },
    {
        "label": "Music",
        "id": "music",
        "type": "radio",
        "default": 0,
        "component": dcc.RadioItems,
        "props": {
            "options": [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
            "inline": True,
        },
    },
    {
        "label": "Volunteering",
        "id": "volunteering",
        "type": "radio",
        "default": 0,
        "component": dcc.RadioItems,
        "props": {
            "options": [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
            "inline": True,
        },
    },
]

def create_form_row(field):
    return dbc.Row(
        [
            dbc.Col(dbc.Label(field["label"], className="fw-bold"), width=4, className="d-flex align-items-center"),
            dbc.Col(
                field["component"](
                    id=field["id"],
                    value=field["default"],
                    **field["props"],
                ),
                width=8,
            ),
        ],
        className="mb-3",
    )

app.layout = dbc.Container(
    [
        # Header
        html.H1(
            "BrightPath's Central ML System",
            className="text-center my-4 fw-bold",
            style={"color": "#f8f9fa"},
        ),
        dcc.Markdown(
            "Where student data becomes insight",
            className="text-center mb-5 text-muted",
        ),
        
        dbc.Card(
            [
                dbc.CardHeader(
                    "Predict Student Grade",
                    className="text-center fw-bold fs-5",
                    style={"backgroundColor": "#343a40", "color": "#f8f9fa"},
                ),
                dbc.CardBody(
                    [create_form_row(field) for field in INPUT_FIELDS],
                    className="p-4",
                ),
                dbc.CardFooter(
                    dbc.Button(
                        "Predict Grade",
                        id="predict-btn",
                        color="primary",
                        className="w-100 fw-bold",
                        size="lg",
                    ),
                    className="p-0",
                ),
            ],
            className="shadow-sm",
            style={
                "maxWidth": "600px",
                "margin": "0 auto",
                "borderRadius": "10px",
                "backgroundColor": "#212529",
            },
        ),
        
        html.Div(
            id="prediction-output",
            className="mt-5 text-center fw-bold",
            style={"fontSize": "1.5rem", "color": "#f8f9fa"},
        ),
    ],
    fluid=True,
    className="py-5",
)

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State(field["id"], "value") for field in INPUT_FIELDS],
    prevent_initial_call=True,
)
def predict(n_clicks, *input_values):
    if not n_clicks:
        return "Enter values and click predict for a prediction"

    if any(value is None for value in input_values):
        return html.Span("Please fill in all fields!", style={"color": "#dc3545"})

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
    try:
        features = pd.DataFrame([dict(zip(feature_names, input_values))])
        prediction = model.predict(features)[0]
        return html.Span(
            f"Predicted Grade: {prediction:.2f}",
            style={"color": "#28a745"},
        )
    except Exception as e:
        return html.Span(f"Error: {str(e)}", style={"color": "#dc3545"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)