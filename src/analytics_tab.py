import pandas as pd
from dash import dcc
import dash_bootstrap_components as dbc
from data import df
from modeling import GradeClassLabel, Predict
import plotly.express as px

gradeClassCounts = df['GradeClass'].value_counts().reset_index()
gradeClassCounts.columns = ['GradeClass', 'count']
gradeClassCounts['GradeClass'] = [GradeClassLabel(int(x)) for x in gradeClassCounts['GradeClass']]

gradeClassPieChart = px.pie(gradeClassCounts, values='count', names='GradeClass', title='Student performance by GradeClass')
gradeClassPieChart.update_layout(
    paper_bgcolor='#212529',
    title_font_color='#f8f9fa',
    legend_title_text='Grades',
    legend=dict(
        font=dict(color='#f8f9fa'),
        bgcolor='#343a40'
    )
)

predictions = Predict(df)

df_annotated = df.copy(deep=True)
df_annotated['GPA_pred'] = predictions['GPA']
df_annotated['GradeClass_pred'] = predictions['GradeClass']

students_of_interest = pd.DataFrame([
    {
        'category': 'Failing',
        'count': df_annotated[df_annotated['GradeClass'] == 4].shape[0],
    },
    {
        'category': 'At risk',
        'count': df_annotated[df_annotated['GPA_pred'] < 2].shape[0],
    },
    {
        'category': 'Watchlist candidate',
        'count': df_annotated[(df_annotated['GradeClass_pred'] > df_annotated['GradeClass']) & (df_annotated['GPA_pred'] >= 2)].shape[0],
    },
    {
        'category': 'Predicted improvement',
        'count': df_annotated[df_annotated['GPA_pred'] > df_annotated['GPA']].shape[0],
    }
])

interestingStudentPieChart = px.pie(students_of_interest, values='count', names='category', title='Students of interest')
interestingStudentPieChart.update_layout(
    paper_bgcolor='#212529',
    title_font_color='#f8f9fa',
    legend_title_text=' Student type',
    legend=dict(
        font=dict(color='#f8f9fa'),
        bgcolor='#343a40'
    )
)

TAB = dbc.Tab(
    [
        dbc.Card(
            [
                dbc.CardHeader(
                    "Student analytics",
                    className="text-center fw-bold fs-5",
                    style={"backgroundColor": "#343a40", "color": "#f8f9fa"},
                ),
                dbc.CardBody(
                    [dbc.Row([
                        dbc.Col(
                            dcc.Graph(figure=gradeClassPieChart),
                            className="shadow-sm",
                            style={
                                "padding": "10px",
                                "margin": "0 auto",
                                "borderRadius": "10px",
                                "backgroundColor": "#212529",
                            }
                        ),
                        dbc.Col(
                            dcc.Graph(figure=interestingStudentPieChart),
                            className="shadow-sm",
                            style={
                                "padding": "10px",
                                "margin": "0 auto",
                                "borderRadius": "10px",
                                "backgroundColor": "#212529",
                            }
                        ),
                    ])],
                    className="p-4",
                ),
            ],
            className="shadow-sm",
            style={
                "maxWidth": "75%",
                "margin": "0 auto",
                "borderRadius": "10px",
                "backgroundColor": "#212529",
            }
        ),
    ],
    label='Analytics',
)
