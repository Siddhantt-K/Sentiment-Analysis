# -*- coding: utf-8 -*-

import pandas as pd
import pickle
import webbrowser
import dash
import plotly.express as px
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer

project_name = 'Sentiment Analysis with Insights'

#def open_browser():
#    webbrowser.open_new("http://127.0.0.1:8050/")
    
def load_model():
    global pickle_model
    global vocab
    global scrappedReviews
    global result

    scrappedReviews = pd.read_csv('scrappedReviews.csv')
    result = pd.read_csv('scrappedReviewsLables.csv')

    
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = pickle.load(file)
    file = open("features.pkl", 'rb') 
    vocab = pickle.load(file)

def check_review(reviewText):
    #load the model
    file = open("pickle_model.pkl", 'rb')
    recreated_model = pickle.load(file)

    #need to covert reviewtext into numeric
    from sklearn.feature_extraction.text import TfidfVectorizer
    vocab = pickle.load(open("features.pkl", 'rb'))
    recreated_vect = TfidfVectorizer(vocabulary = vocab) #created model/object
    reviewText_vectorized = recreated_vect.fit_transform([reviewText]) #we are converting text to numeric (into 25359 features)
    return recreated_model.predict(reviewText_vectorized) #returning

def plot():
    pie_r = result.groupby(["lables"])["lables"].count()
    pie_result = pd.DataFrame(pie_r)
    pie_result["names"] = ["Negative","Positive"]
    pie_result.rename(columns = {'lables':'Count'}, inplace = True)
    print(pie_result)
    plot.pie_fig = go.Figure(px.pie(pie_result, values='Count',names='names'))
    plot.pie_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Courier New, monospace",
            size=22,
        )
    )



app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])

project_name = "Sentiment Analysis with Insights"

def create_app_ui(): 
    global project_name
    main_layout = html.Div([

                                html.Br(),
                                html.H1(id = 'heading', children = project_name, className = 'text-secondary font-weight-bolder display-3 mb-4'),  
                                html.Hr(),                                
                                html.H3("Pie Chart",className="bg-light font-weight-bold text-center"),
                                dcc.Graph(id='pie', figure=plot.pie_fig, className="text-center"),
                                dcc.Loading( type="default"),
                                html.H3("Word Cloud",className="bg-light font-weight-bold text-center"),
                                html.Img(src="https://raw.githubusercontent.com/Siddhantt-K/wordcloud/master/wordcloud_image.png", className="img-responsive img-thumbnail text-center", style={'height':'30%', 'width':'100%'}),                                
                                html.Br(),
                                html.Br(),
                                html.H3("Type your review to check sentiments",className="bg-light font-weight-bold text-center",style={'margin-top':'10px'}),
                                dbc.Textarea(id = 'textarea', placeholder="Enter the Review"),
                                html.Div([
                                        dbc.Button("Submit", color="dark", className="mt-2 mb-1", id = 'button', style = {'width': '80px'})
                                    ],className="text-center"),
                                

                                html.H4(id = 'result', children = 'Review Result:', className="text-secondary font-weight-normal text-center"),
                                html.Br(),
                                html.Br(),
                                html.H3("You may choose any random review from the below dropdown",
                                                       className = 'bg-light font-weight-bold text-center'
                                                       ),
                                        
                                dcc.Dropdown(id = 'my-dpdn',
                                                             placeholder = 'Select a Review',
                                                             options=[{'label': i[:100], 'value': i} for i in scrappedReviews.reviews],
                                                             value = scrappedReviews.reviews[0],
                                                             multi = False,
                                                             #style = {'margin-bottom': '30px', 'marginTop': 25}
                                                             ), 
                                html.Div([
                                        dbc.Button("Submit", color="dark", className="mt-2 mb-1", id = 'dpdw-button', style = {'width': '80px'})
                                    ],className="text-center"),                                                               
                                html.H4(id = 'result1', children = 'none',className="text-center")                            
                                                                                
                                ], className = "container")
    return main_layout


@app.callback(
    Output('result', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )    
def update_app_ui(n_clicks, textarea):
    result_list = check_review(textarea)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")
    

@app.callback(
    Output('result1', 'children'),
    [
    Input('dpdw-button', 'n_clicks')
    ],
    [
     State('my-dpdn', 'value')
    ]
    )
def update_dropdown(n_clicks, value):
    result_list = check_review(value)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")

def main():
    global app
    global project_name
	
    load_model()
    plot()
    print('Start of project')
#    open_browser()
    app.title = project_name
    app.layout = create_app_ui()
    app.run_server(host = '0.0.0.0', port = 8080)

    app = None
    project_name = None    

if __name__=='__main__':
    main()    


