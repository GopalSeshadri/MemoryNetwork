import numpy as np
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go
import dash
import dash_core_components as core
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table

from main import Main

app = dash.Dash()

app.layout = html.Div([
    html.Div([
        html.H1('Answering Single/Double Fact Story', style = {'fontFamily' : 'Helvetica',
                                                        'textAlign' : 'center',
                                                        'width' : '100%'})
    ], style = {'display' : 'flex'}),

    html.Div([

        html.Div([
            html.Button(id = 'fetch-button', children = 'Fetch Story',
                        style = {'width' : '100px',
                                'height' : '50px'})
        ], style = {'width' : '100%',
                    'paddingLeft' : '20px',
                    'display' : 'inline-block',
                    'float' : 'left'}),

        html.Div([

            core.Textarea(id = 'story-area',
                placeholder = 'Generating Story ...',
                style = {'width': '600px',
                        'height' : '200px',
                        'display' : 'inline-block',
                        'float' : 'left'}
            )
        ], style = {'width' : '100%',
                    'display' : 'flex',
                    'paddingTop' : '20px'}),

        html.Div([

            core.Textarea(id = 'question-area',
                placeholder = 'Generating Question ...',
                style = {'width': '600px',
                        'height' : '30px',
                        'display' : 'flex',
                        'float' : 'left'}
            )
        ], style = {'width' : '100%',
                    'display' : 'flex',
                    'paddingTop' : '20px'}),

        html.Div([

            core.Textarea(id = 'predicted-area',
                placeholder = 'Generating Predicted Answer ...',
                style = {'width': '600px',
                        'height' : '30px',
                        'display' : 'inline-block',
                        'float' : 'left'}
            )
        ], style = {'width' : '100%',
                    'display' : 'flex'}),

        html.Div([

            core.Textarea(id = 'correct-area',
                placeholder = 'Generating Correct Answer ...',
                style = {'width': '600px',
                        'height' : '30px',
                        'display' : 'inline-block',
                        'float' : 'left'}
            )
        ], style = {'width' : '100%',
                    'display' : 'flex'})

    ], style = {'width' : '40%',
                'height' : '80%',
                'display' : 'inline-block',
                'float' : 'left',
                'paddingLeft' : '50px'}),

    html.Div([
        dash_table.DataTable(
            id = 'weights-table',
            columns = [{'name' : i, 'id' : i} for i in ['Weights_1', 'Weights_2', 'Sentence']])
    ], style = {'width' : '25%',
                'height' : '600px',
                'display' : 'inline-block',
                'float' : 'left',
                'paddingLeft' : '50px'})
], style = {'fontFamily' : 'Helvetica',
            'width' : '100%',
            'height' : '100%'})


@app.callback([Output(component_id = 'story-area', component_property = 'value'),
            Output(component_id = 'question-area', component_property = 'value'),
            Output(component_id = 'predicted-area', component_property = 'value'),
            Output(component_id = 'correct-area', component_property = 'value'),
            Output(component_id = 'weights-table', component_property= 'data'),
            Output(component_id = 'weights-table', component_property = 'style_data_conditional')],
            [Input(component_id = 'fetch-button', component_property = 'n_clicks')])
def affectTextboxAndTable(n_clicks):
    '''
    This is a callback function. It takes as input the number of clicks and returns the data for story, question and answer text areas and the weights table.

    Parameters:
    n_clicks (int) : The number of times the fetch button was clicked

    Returns:
    values for story-area, question-area, predicted-area and correct-area text areas.
    weights_table (dict) : The data with weights1, weights2 and sentences for weights DataTable
    style_data_conditional (list) : A list of dictionary for styling the rows of weights DataTable
    '''
    choice_dict = {0 : 'single', 1 : 'double'}

    choice = np.random.choice(2)
    story, question, correct_answer, weights1, weights2, predicted_answer = Main.generateAnswer(choice_dict[choice])

    weights1 = [round(w, 2) for w in weights1.tolist()[:len(story)]]
    weights2 = [round(w, 2) for w in weights2.tolist()[:len(story)]]
    weights_dict = {'Weights_1' : weights1, 'Weights_2' : weights2, 'Sentence' : story}
    weights_table = pd.DataFrame(weights_dict).to_dict('records')

    style_data_conditional = [{
        'if': {'filter_query': '{Weights_2} > 0.5'},
        'backgroundColor': '#3D9970',
        'fontWeight' : 'bold',
        'color': 'white'
    },
    {
        "if": {'filter_query': '{Weights_2} <= 0.5'},
        'backgroundColor': '#ffffff',
        'color': 'black'
    }]

    return '\n'.join(story), 'The Question is : ' + question, \
        'The Predicted Answer is : ' + predicted_answer, 'The Correct Answer is : ' + correct_answer, \
        weights_table, style_data_conditional



if __name__ == '__main__':
    app.run_server()
