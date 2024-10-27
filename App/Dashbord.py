import dash
from dash import dcc, html
import plotly.graph_objects as go
import requests
from dash.dependencies import Input, Output

# External stylesheets for the dashboard
external_stylesheets = ['https://fonts.googleapis.com/css2?family=Lobster&display=swap']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Layout of the dashboard
app.layout = html.Div([
    html.Div([
        html.H1("Fraud Insights Dashboard", style={
            'fontFamily': 'Lobster, cursive',
            'textAlign': 'center',
            'color': '#fff',
            'marginBottom': '30px',
            'marginTop': '0px',
            'padding': '20px',
            'backgroundColor': '#4A90E2',
            'borderRadius': '10px'
        }),
    ], style={'marginBottom': '20px'}),
    
    html.Div(id='summary-boxes'),

    html.Div([
        html.Div([
            dcc.Graph(id='fraud-trends', style={'width': '100%'})
        ], style={
            'padding': '20px',
            'backgroundColor': '#f9f9f9',
            'border': '1px solid #ccc',
            'borderRadius': '10px',
            'maxWidth': '950px',
            'margin': '0 auto',
            'textAlign': 'center'
        })
    ], style={'textAlign': 'center', 'width': '100%'}),

    html.Div([
        dcc.Graph(id='fraud-map', style={'width': '100%'})
    ], style={
        'padding': '20px',
        'backgroundColor': '#f9f9f9',
        'border': '1px solid #ccc',
        'borderRadius': '10px',
        'maxWidth': '950px',
        'textAlign': 'center',
        'margin': '40px auto'
    }),
    # New section for the bar chart
     # New section for two separate bar charts
    html.Div([
        html.Div([
            dcc.Graph(id='fraud-by-device-chart', style={'width': '100%'})
        ], style={'flex': '1', 'padding': '10px'}),
        
        html.Div([
            dcc.Graph(id='fraud-by-browser-chart', style={'width': '100%'})
        ], style={'flex': '1', 'padding': '10px'}),
    ], style={'display': 'flex','backgroundColor': '#f9f9f9','maxWidth': '950px', 'justifyContent': 'space-between', 'flexWrap': 'wrap', 'padding': '20px','textAlign': 'center','margin': '40px auto'})
])


@app.callback(
    Output('summary-boxes', 'children'),
    Input('summary-boxes', 'id')
)
def update_summary_boxes(_):
    try:
        response = requests.get('http://127.0.0.1:5000/api/summary')
        response.raise_for_status()  # Raise an error for bad responses
        summary = response.json()
    except requests.RequestException as e:
        return html.Div(f"Error retrieving summary data: {e}")

    return html.Div([
        html.Div("Fraud Statistics Summary", style={
            'fontSize': '30px',
            'textAlign': 'center',
            'width': '60%',
            'margin': '0 auto',
            'color': '#333',
            'fontFamily': 'Lobster, cursive'
        }),

        html.Div([
            html.Div([
                html.Div("Total Transactions", style={'fontSize': '25px', 'color': '#333', 'marginBottom': '5px'}),
                html.Div(f"{summary['total_transactions']}", style={'fontSize': '24px', 'color': '#4CAF50'})
            ], style={'width': '250px', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'border': '1px solid #ccc', 'borderRadius': '10px', 'textAlign': 'center'}),

            html.Div([
                html.Div("Total Fraud Cases", style={'fontSize': '25px', 'color': '#333', 'marginBottom': '5px'}),
                html.Div(f"{summary['total_fraud_cases']}", style={'fontSize': '24px', 'color': '#FF5733'})
            ], style={'width': '250px', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'border': '1px solid #ccc', 'borderRadius': '10px', 'textAlign': 'center'}),

            html.Div([
                html.Div("Fraud Percentage", style={'fontSize': '25px', 'color': '#333', 'marginBottom': '5px'}),
                html.Div(f"{summary['fraud_percentage']:.2f}%", style={'fontSize': '24px', 'color': '#2196F3'})
            ], style={'width': '250px', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'border': '1px solid #ccc', 'borderRadius': '10px', 'textAlign': 'center'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '10px', 'flexWrap': 'wrap', 'maxWidth': '950px', 'margin': '0 auto'})
    ], style={'textAlign': 'center', 'padding': '20px', 'borderRadius': '10px'})

@app.callback(
    Output('fraud-trends', 'figure'),
    Input('summary-boxes', 'children')
)
def update_fraud_trends(_):
    try:
        response = requests.get('http://127.0.0.1:5000/api/fraud_trends')
        response.raise_for_status()
        trends = response.json()
    except requests.RequestException as e:
        return go.Figure(data=[])

    dates = [entry['purchase_time'] for entry in trends]
    fraud_cases = [entry['fraud_cases'] for entry in trends]

    figure = go.Figure(data=go.Scatter(
        x=dates,
        y=fraud_cases,
        mode='lines+markers',
        name='Fraud Cases',
        line=dict(color='blue'),
        marker=dict(size=5)
    ))

    figure.update_layout(
        title=dict(
            text='Fraud Cases Over Time',
            font=dict(family='Lobster, cursive', size=24, color='black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Date',
        yaxis_title='Number of Fraud Cases',
        xaxis=dict(tickformat='%Y-%m'),
        hovermode='closest',
        margin=dict(l=50, r=50, t=40, b=40)
    )

    return figure

@app.callback(
    Output('fraud-map', 'figure'),
    Input('summary-boxes', 'children')
)
def update_fraud_map(_):
    try:
        country_data = requests.get('http://127.0.0.1:5000/api/fraud_by_country')
        country_data.raise_for_status()
        countries = [entry['country'] for entry in country_data.json()]
        fraud_cases = [entry['fraud_cases'] for entry in country_data.json()]
    except requests.RequestException as e:
        return go.Figure(data=[])

    figure = go.Figure(data=go.Choropleth(
        z=fraud_cases,
        locations=countries,
        locationmode='country names',
        colorscale='Viridis',
        colorbar=dict(title='Fraud Cases'),
        hoverinfo='location+z',
    ))

    figure.update_layout(
        title=dict(
            text='Fraud Cases by Country',
            font=dict(family='Lobster, cursive', size=24, color='black'),
            x=0.5,
            xanchor='center'
        ),
        geo=dict(showframe=False, projection_type='natural earth'),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return figure
# Callback for fraud cases by device
@app.callback(
    Output('fraud-by-device-chart', 'figure'),
    Input('summary-boxes', 'children')
)
def update_fraud_by_device_chart(_):
    try:
        response = requests.get('http://127.0.0.1:5000/api/fraud_by_device_browser')
        response.raise_for_status()
        data = response.json()

        device_fraud = {}
        for entry in data:
            device_id = entry['device_id']
            fraud_cases = entry['fraud_cases']
            device_fraud[device_id] = device_fraud.get(device_id, 0) + fraud_cases

        devices = list(device_fraud.keys())
        fraud_counts = list(device_fraud.values())

    except requests.RequestException as e:
        return go.Figure(data=[])

    figure = go.Figure(data=go.Bar(
        x=devices,
        y=fraud_counts,
        marker=dict(color='lightcoral')
    ))

    figure.update_layout(
        title=dict(
            text='Top Five Fraud Cases by Device',
            font=dict(family='Lobster, cursive', size=24, color='black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Device ID',
        yaxis_title='Number of Fraud Cases',
        margin=dict(l=10, r=10, t=40, b=40)
    )

    return figure

# Callback for fraud cases by browser
@app.callback(
    Output('fraud-by-browser-chart', 'figure'),
    Input('summary-boxes', 'children')
)
def update_fraud_by_browser_chart(_):
    try:
        response = requests.get('http://127.0.0.1:5000/api/fraud_by_device_browser')
        response.raise_for_status()
        data = response.json()

        browser_fraud = {}
        for entry in data:
            browser = entry['browser']
            fraud_cases = entry['fraud_cases']
            browser_fraud[browser] = browser_fraud.get(browser, 0) + fraud_cases

        browsers = list(browser_fraud.keys())
        fraud_counts = list(browser_fraud.values())

    except requests.RequestException as e:
        return go.Figure(data=[])

    figure = go.Figure(data=go.Bar(
        x=browsers,
        y=fraud_counts,
        marker=dict(color='skyblue')
    ))

    figure.update_layout(
        title=dict(
            text='Top Five Fraud Cases by Browser',
            font=dict(family='Lobster, cursive', size=24, color='black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Browser',
        yaxis_title='Number of Fraud Cases',
        margin=dict(l=10, r=10, t=40, b=40)
    )

    return figure


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)