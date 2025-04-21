import dash_bootstrap_components as dbc
from prediction_tab import TAB as pred_tab
from analytics_tab import TAB as anal_tab
from dash import html, dcc
from app_var import app

server = app.server

app.layout = dbc.Container(
    [
        # Header
        html.H1(
            "BrightPath's Central ML System",
            className="text-center my-4 fw-bold",
            style={"color": "#f8f9fa"},
        ),

        dcc.Markdown(
            "### Where student data becomes insight",
            className="text-center mb-5 text-muted",
        ),

        dbc.Tabs([
                anal_tab,
                pred_tab,
            ],
            style={
                "maxWidth": "75%",
                "margin": "0 auto",
            }
        ),
    ],
    fluid=True,
    className="py-5"
)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
