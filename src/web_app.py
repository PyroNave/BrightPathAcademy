import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import joblib
import pandas as pd

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
model = joblib.load('../artifacts/random_forest_regressor.joblib')

app.layout = dbc.Container([
    html.H1("BrightPath's Central ML system", className="text-center my-4"),
    dcc.Markdown("Where student data becomes insight", className="text-center mb-4"),
    
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.CardHeader([
                    dbc.Label("Predict with the model"),
                ]),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label("Age"), width=4),
                dbc.Col(dcc.Input(id='age', type='number', value=18), width=4),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label("Gender"), width=4),
                dbc.Col(dcc.Dropdown(
                    id='gender',
                    options=[
                        {'label': 'Male', 'value': 0},
                        {'label': 'Female', 'value': 1},
                    ],
                    value=0,
                )),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label("Ethnicity"), width=4),
                dbc.Col(dcc.Dropdown(
                    id='ethnicity',
                    options=[
                        {'label': 'Caucasian', 'value': 0},
                        {'label': 'African American', 'value': 1},
                        {'label': 'Asian', 'value': 2},
                        {'label': 'Other', 'value': 3},
                    ],
                    value=0,
                )),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label("Parental Education"), width=4),
                dbc.Col(dcc.Dropdown(
                    id='parent-edu',
                    options=[
                        {'label': 'None', 'value': 0},
                        {'label': 'High School', 'value': 1},
                        {'label': 'Some College', 'value': 2},
                        {'label': 'Bachelor\'s', 'value': 3},
                        {'label': 'Higher Study', 'value': 4},
                    ],
                    value=0,
                )),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label("Study Time Weekly (hrs)"), width=4),
                dbc.Col(dcc.Input(id='study-time', type='number', value=0)),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label("Absences"), width=4),
                dbc.Col(dcc.Input(id='absences', type='number', value=0)),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label("Tutoring"), width=4),
                dbc.Col(dcc.RadioItems(
                    id='tutoring',
                    options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                    inline=True,
                    value=0,
                )),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label("Parental Support:"), width=4),
                dbc.Col(dcc.RadioItems(
                    id='parent-support',
                    options=[
                        {'label': 'None', 'value': 0},
                        {'label': 'Low', 'value': 1},
                        {'label': 'Moderate', 'value': 2},
                        {'label': 'High', 'value': 3},
                        {'label': 'Very High', 'value': 4},
                    ],
                    value=0,
                )),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label("Extracurricular"), width=4),
                dbc.Col(dcc.RadioItems(
                    id='extracurricular',
                    options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                    inline=True,
                    value=0,
                )),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label("Sports"), width=4),
                dbc.Col(dcc.RadioItems(
                    id='sports',
                    options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                    inline=True,
                    value=0,
                )),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label("Music"), width=4),
                dbc.Col(dcc.RadioItems(
                    id='music',
                    options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                    inline=True,
                    value=0,
                )),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label("Volunteering"), width=4),
                dbc.Col(dcc.RadioItems(
                    id='volunteering',
                    options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                    inline=True,
                    value=0,
                )),
            ]),
        ]),

        dbc.Button("Predict Grade", id="predict-btn", color="primary", className="w-100"),
    ], style={'display': 'grid', 'gap': '10px', 'maxWidth': '400px', 'margin': '0 auto'}),

    html.Div(id='prediction-output', style={'marginTop': 20, 'fontSize': 24, 'textAlign': 'center'})
], fluid=True)

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('age', 'value'),
    State('gender', 'value'),
    State('ethnicity', 'value'),
    State('parent-edu', 'value'),
    State('study-time', 'value'),
    State('absences', 'value'),
    State('tutoring', 'value'),
    State('parent-support', 'value'),
    State('extracurricular', 'value'),
    State('sports', 'value'),
    State('music', 'value'),
    State('volunteering', 'value'),
    prevent_initial_call=True,
)
def predict(n_clicks, age, gender, ethnicity, parent_edu, study_time, absences, tutoring, parent_support, extracurricular, sports, music, volunteering):
    if n_clicks == 0:
        return "Enter values and click predict for a prediction"

    values = [age, gender, ethnicity, parent_edu, study_time, absences, tutoring, parent_support, extracurricular, sports, music, volunteering]

    if None in values:
        return "Please enter all values!"

    try:
        features = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Ethnicity': ethnicity,
            'ParentalEducation': parent_edu,
            'StudyTimeWeekly': study_time,
            'Absences': absences,
            'Tutoring': tutoring,
            'ParentalSupport': parent_support,
            'Extracurricular': extracurricular,
            'Sports': sports,
            'Music': music,
            'Volunteering': volunteering
        }])
        prediction = model.predict(features)
        return f"Predicted Grade: {prediction[0]:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
