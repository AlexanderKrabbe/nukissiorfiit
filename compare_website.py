import time
import datetime
import locale
import statistics

import plotly
import plotly.offline as py
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import pandas as pd

from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly


colors = ["#1E88E5","#FFC107","#004D40","#D81B60"]
st = time.time()



""" READ IN DATA """
df_read = pd.read_csv('data.csv', sep=',', dtype={'City': str}, index_col=0, parse_dates=['Date', 'Periode'])
df = df_read
df['B'] = df[['Brændolie forbrug','Forbrug brændolie']].sum(axis=1)
df['Brændolie forbrug'] = df['B']
df['Forbrug brændolie'] = df['B']


""" MAKE DAILY DATAFRAME """
df_daily = df.drop('Periode', axis = 1)
df_daily = df_daily[df_daily['Date'].notnull()]
df_daily = df_daily.sort_values(by=['City'])


""" MAKE MONTHLY DATAFRAME """
df_monthly = df.drop('Date', axis = 1)#.set_index('Periode')
df_monthly = df_monthly[df_monthly['Periode'].notnull()]
df_monthly = df_monthly.sort_values(by=['City'])

""" LOCATION VARIABLE """
tid = 'Date' #'Periode' #


city_start = '013 Narsamijit'
df_test = df
if tid == 'Date':
    df = df_daily
    x_name='Dag'
elif tid == 'Periode':
    df = df_monthly
    x_name='Måned'


col1 = 'Elvirkningsgrad'
col2 = 'Brændolie forbrug'
col3 = 'Produktion total'
col4 = 'Max. belasting'


col, enhed = col2, ' [liter olie]'
observationer = ['Produktion total', 'Brændolie forbrug', 'Elvirkningsgrad', 'Gram pr. kWh']


max_value = 1400
min_value = 1

fra_år = 0

titel = col+' for '+str(fra_år)+' for '

""" DATA """

meta_data = {}
visibles = []
n = 0

liste_over_byer = df['City'].unique().tolist()
""" First we choose tidsopløsning """
""" Dernæst vælges byen """
""" Dernæst vælges type """

city = city_start

df_plot = df[df['City'] == city][observationer + [tid, 'City']]
df_plot = df_plot.sort_values(by=[tid])

rolling_mean = 1



print(df_plot[df_plot['City'] == city].dropna(how='all', axis = 1).columns.tolist())



""" PLOT THE NEXT BIG THING ASGERS PLOT"""
df_plot = df_daily
tid = 'Date'
df_plot = df_plot[(df_plot['City'] == '013 Narsamijit') & (df_plot[tid].between('2020-01-01','2020-12-31'))][['Produktion til byen','Eget forbrug','Vejlys','Produktion total',tid]]
df_plot = df_plot.sort_values(by=tid)


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.update_layout(go.Layout(
        barmode='stack',
        bargap=0.15,
        plot_bgcolor="#FFF",#'rgba(55,255,255,0)',
 #       paper_bgcolor='rgba(255,255,255,0)',
    ),
    )



fig.add_trace(
            go.Bar(
                x =df_plot[tid],
                y = df_plot['Produktion til byen']/df_plot['Produktion total'],
                name='Produktion til byen',
                #line=dict(color=colors[0])
                ),
        #color="#ff7f0e",
        #secondary_y=False,
        )
fig.add_trace(
            go.Bar(
                x =df_plot[tid],
                y = df_plot['Eget forbrug']/df_plot['Produktion total'],
                name='Eget forbrug',
                ),
        )
fig.add_trace(
            go.Bar(
                x =df_plot[tid],
                y = df_plot['Vejlys']/df_plot['Produktion total'],
                name='Vejlys',
                #line=dict(color=colors[2])
                ),
        #color="#ff7f0e",
        #secondary_y=False,
        )
fig.add_trace(go.Scatter(
    x=df_plot[tid],
    y=df_plot['Produktion total'],
    name='Total [kWh]'),
    #line=dict(color=colors[3]),
    #color="#ff7f0e",
    secondary_y=True,)

#fig.write_html('box_af_vejlys.html')

print(time.time()-st)

""" Here the website begins """

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


app = dash.Dash()

""" LAYOUT """

app.layout = html.Div([
    html.H1(children='Nukissiofiit bygdedata'),

    dcc.Dropdown(
        options=[
            {'label': i, 'value': i} for i in df['City'].unique()
        ],
        value='013 Narsarmijit',
        #multi=True,
        id='input_city',
        ),
    dcc.Dropdown(
        options=[
            {'label': i, 'value': i} for i in df.columns.to_list()
        ],
        value='Brændolie forbrug',
        id='input_kat',
        ),

 #   dcc.Dropdown(
 #       options=[
 #           {'label': '2020', 'value': '2020'}# for i in df[''].unique()print(df[tid].dt.strftime("%y").unique().tolist())
 #       ],
 #       value='2020',
        #multi=True,
 #       id='input_year',
  #      ),

    dcc.Graph(id='graf_forecast'),
    dcc.Graph(id='graf_forecast_components'),
#    dcc.Graph(id='grafen'),
#    dcc.Graph(id='grafen_2'),
#    dcc.Graph(id='fig_gram'),
#    dcc.Graph(id='fig_distrikt'),
#    dcc.Graph(id='fig_box'),
#    dcc.Graph(id='fig_samlet_overblik'),
    ])

"""CALL BACKS"""
@app.callback(
    [Output(component_id='graf_forecast', component_property='figure'),
    Output(component_id='graf_forecast_components', component_property='figure')],
    [Input(component_id='input_city', component_property='value'),
    Input(component_id='input_kat', component_property='value')]
)
def update_forecast(input_city, input_kat):
    #print(input_city, input_year)
    if input_city:
        city = input_city
        print('input city: ', input_city)
    else:
        city = '013 Narsarmijit'
    if input_kat:
        kategori = input_kat
        print('input kat: ', input_kat)
    else:
        kategori = 'Brændolie forbrug'

    df_forecast = pd.DataFrame(columns=['ds','y'])
    df_temp = df[df['City'] == city][[tid, kategori]]
    df_temp = df_temp.sort_values(by=[tid])

    #df_temp = df_temp[(df_temp['Elvirkningsgrad'] > 0)& (df_temp['Elvirkningsgrad'] < 105)]
    #df_temp = df_temp[(df_temp['Gram pr. kWh'] > 0)& (df_temp['Gram pr. kWh'] < 555)]
    #df_temp = df_temp[(df_temp['Brændolie forbrug'] > 0)& (df_temp['Brændolie forbrug'] < 2500)]

    
    df_forecast[['ds','y']] = df_temp[[tid, kategori]]

    q1 = df_forecast['y'].quantile(0.25)
    q3 = df_forecast['y'].quantile(0.75)
    iqr = q3-q1

    min_value = q1-1.5*iqr
    max_value = q3+1.5*iqr

    df_forecast = df_forecast[(df_forecast['y'] > min_value) & (df_forecast['y'] < max_value)]


    df_forecast = df_forecast[df_forecast['ds'].notna()]

    m = Prophet()
    m.fit(df_forecast)

    future = m.make_future_dataframe(periods=365)

    forecast = m.predict(future)

    fig1 = m.plot(forecast)

    fig2 = m.plot_components(forecast)

    fig_comp = plot_components_plotly(m, forecast)

    fig = plot_plotly(m, forecast)

    return [fig, fig_comp]


##@app.callback(
##    Output(component_id='graf_forecast_components', component_property='figure'),
##    [Input(component_id='input_city', component_property='value'),
##    Input(component_id='input_kat', component_property='value')]
##)
##def update_forecast(input_city, input_kat):
##    #print(input_city, input_year)
##    if input_city:
##        city = input_city
##        print('input city: ', input_city)
##    else:
##        city = '013 Narsarmijit'
##    if input_kat:
##        kategori = input_kat
##        print('input kat: ', input_kat)
##    else:
##        kategori = 'Brændolie forbrug'
##
##    df_forecast = pd.DataFrame(columns=['ds','y'])
##    df_temp = df[df['City'] == city][[tid, kategori]]
##    df_temp = df_temp.sort_values(by=[tid])
##
##    #df_temp = df_temp[(df_temp['Elvirkningsgrad'] > 0)& (df_temp['Elvirkningsgrad'] < 105)]
##    #df_temp = df_temp[(df_temp['Gram pr. kWh'] > 0)& (df_temp['Gram pr. kWh'] < 555)]
##    #df_temp = df_temp[(df_temp['Brændolie forbrug'] > 0)& (df_temp['Brændolie forbrug'] < 2500)]
##
##    
##    df_forecast[['ds','y']] = df_temp[[tid, kategori]]
##
##    q1 = df_forecast['y'].quantile(0.25)
##    q3 = df_forecast['y'].quantile(0.75)
##    iqr = q3-q1
##
##    min_value = q1-1.5*iqr
##    max_value = q3+1.5*iqr
##
##    df_forecast = df_forecast[(df_forecast['y'] > min_value) & (df_forecast['y'] < max_value)]
##
##
##    df_forecast = df_forecast[df_forecast['ds'].notna()]
##
##    m = Prophet()
##    m.fit(df_forecast)
##
##    future = m.make_future_dataframe(periods=365)
##
##    forecast = m.predict(future)
##
##    fig1 = m.plot(forecast)
##
##    fig2 = m.plot_components(forecast)
##
##    fig = plot_plotly(m, forecast)
##
##    return fig2

#@app.callback(
#    Output(component_id='fig_samlet_overblik', component_property='figure'),
#    Input(component_id='input_city', component_property='value'),
#    Input(component_id='input_year', component_property='value')
#)
#def update_box(input_city, input_year):
#    #print(input_city, input_year)
#    df = df_daily
#    df_temp = df[(df['Elvirkningsgrad'] > 0)& (df['Elvirkningsgrad'] < 105)]
#    df_temp = df_temp[(df_temp['Gram pr. kWh'] > 0)& (df_temp['Gram pr. kWh'] < 555)]
#    df_temp = df_temp[(df_temp['Brændolie forbrug'] > 0)& (df_temp['Brændolie forbrug'] < 2500)]
#    df_temp = df_temp[(df_temp['Date'].isin(pd.date_range(start='1/1/2020', end='31/12/2020')))]
#    df_temp['median gram olie'] = df_temp.groupby('City')['Gram pr. kWh'].transform('median')
#    df_temp = df_temp.loc[df_temp['median gram olie'].sort_values(ascending=False).index]
#
#    fig_box = px.violin(df_temp, x="City", y="Gram pr. kWh", color ='City', box=True, hover_data=df.columns)
#    fig_box.update_layout(title = 'Violinplot af "Gramolie" for alle byer' )
#    return fig_box
#
#''' Her kommer der en graf  '''
#@app.callback(
#    Output(component_id='fig_box', component_property='figure'),
#    Input(component_id='input_city', component_property='value'),
#    Input(component_id='input_year', component_property='value')
#)
#def update_box(input_city, input_year):
#    print(input_city, input_year)
#    df = df_daily
#    df_temp = df[(df['Elvirkningsgrad'] > 0)& (df['Elvirkningsgrad'] < 105)]
#    df_temp = df_temp[(df_temp['Gram pr. kWh'] > 0)& (df_temp['Gram pr. kWh'] < 555)]
#    df_temp = df_temp[(df_temp['Brændolie forbrug'] > 0)& (df_temp['Brændolie forbrug'] < 2500)]
#    df_temp = df_temp[(df_temp['Date'].isin(pd.date_range(start='1/1/2020', end='31/12/2020')))]
#
#    fig_box = px.violin(df_temp[df_temp['City'].isin(input_city)], x="City", y= "Gram pr. kWh",color ='City', box=True, points='all', hover_data=df.columns)
#    fig_box.update_layout(title = 'Violinplot af "Gramolie"' )
#    return fig_box
#
#@app.callback(
#    Output(component_id='fig_gram', component_property='figure'),
#    Input(component_id='input_city', component_property='value')
#)
#def update_gram_olie(input_city):
#
#    df = df_daily
#    df_temp = df[(df['Elvirkningsgrad'] > 0)& (df['Elvirkningsgrad'] < 105)]
#    df_temp = df_temp[(df_temp['Gram pr. kWh'] > 0)& (df_temp['Gram pr. kWh'] < 555)]
#    df_temp = df_temp[(df_temp['Brændolie forbrug'] > 0)& (df_temp['Brændolie forbrug'] < 2500)]
#    df_temp = df_temp[(df_temp['Date'].isin(pd.date_range(start='1/1/2020', end='31/12/2020')))]
#    #df_temp['year'] = df_temp['Date'].dt.year.astype(str)
#
#    fig_gram = px.scatter(df_temp[df_temp['City'].isin(input_city)], x="Brændolie forbrug", y= "Gram pr. kWh",color ='City', marginal_x="violin", marginal_y="violin")
#    fig_gram.update_layout(title = 'Korrelation af "Gramolie" og "Elvirkningrad"')
#
#    return fig_gram
#
#@app.callback(
#    Output(component_id='fig_distrikt', component_property='figure'),
#    Input(component_id='input_city', component_property='value')
#)
#def update_distrikt(input_city):
#
#    df = df_daily
#    df_temp = df[(df['Elvirkningsgrad'] > 0)& (df['Elvirkningsgrad'] < 105)]
#    df_temp = df_temp[(df_temp['Gram pr. kWh'] > 0)& (df_temp['Gram pr. kWh'] < 555)]
#    df_temp = df_temp[(df_temp['Brændolie forbrug'] > 0)& (df_temp['Brændolie forbrug'] < 2500)]
#    df_temp = df_temp[(df_temp['Date'].isin(pd.date_range(start='1/1/2020', end='31/12/2020')))]
#
#    fig_distrikt = px.scatter(df_temp, x="Brændolie forbrug", y= "Gram pr. kWh", color ='Distrikt', marginal_x="violin", marginal_y="violin")
#    #fig_distrikt = px.scatter(df_temp.groupby(['Distrikt','Date']).mean().reset_index(), x="Elvirkningsgrad", y= "Gram pr. kWh", color ='Distrikt', marginal_x="violin", marginal_y="violin")
#    fig_distrikt.update_layout(title = 'Korrelation af "Gramolie" og "Brændolie forbrug"')
#
#    fig2 = px.scatter(df_temp[df_temp['City'].isin(input_city)], x="Brændolie forbrug", y= "Gram pr. kWh",color ='City', marginal_x="violin", marginal_y="violin")
#
#    for i,v in enumerate(fig2.data):
#        fig_distrikt.add_trace(v)
#
#    return fig_distrikt
#
#@app.callback(
#    Output(component_id='grafen', component_property='figure'),
#    Input(component_id='input_city', component_property='value')
#)
#def update_output_div(input_city):
#
#    if input_city:
#        city=input_city[-1]
#        print(city)
#    else:
#        city = '013 Narsamijit'
#
#    df_plot = df[df['City'] == city][observationer + [tid, 'City']]
#    df_plot = df_plot.sort_values(by=[tid])
#    rolling_mean = 14
#
#    mean_el = df_plot[['Produktion total', 'City']][(df_plot['City'] == city) & (df_plot['Produktion total'] > 0) & (df_plot['Produktion total'] < 50000000)].mean()
#    std_el = df_plot[['Produktion total', 'City']][(df_plot['City'] == city) & (df_plot['Produktion total'] > 0) & (df_plot['Produktion total'] < 50000000)].std()
#
#    years = [y for y in df_plot[tid].dt.strftime('%Y').unique().tolist()]# if int(y) >= fra_år]
#
#    X = df_plot[[tid]][(df_plot['City'] == city)][tid]
#    fig = go.Figure()
#
#
#    fig = make_subplots(specs=[[{"secondary_y": True}]])
#    for year in years:
#        fig.add_trace(
#        go.Scatter(
#            #x = X,
#            y =
#df_plot[df_plot[tid].between(str(year)+'-01-01',str(year)+'-12-31')][observationer[0]].rolling(rolling_mean).mean(),
#            #y = df_plot[[observationer[0], 'City']][(df_plot['City'] == city) & (df_plot[observationer[0]] > mean_el[0]-3*std_el[0]) & (df_plot[observationer[0]] < mean_el[0]+3*std_el[0])][observationer[0]].rolling(rolling_mean).mean(),
#            name=year+' '+observationer[0],
#            line=dict(color=colors[0]),
#            opacity=0.2,
#             ),
#        #color="#1f77b4"
#        #secondary_y=False,   
#        )
#
#        fig.add_trace(
#            go.Scatter(
#                #x =X,
#                #y =df_plot[[observationer[1] , 'City']][(df_plot['City'] == city)][observationer[1]].rolling(rolling_mean).mean(),
#                y = df_plot[df_plot[tid].between(str(year)+'-01-01',str(year)+'-12-31')][observationer[1]],
#                name=year+' '+observationer[1],
#                yaxis = "y2",
#                opacity=0.2,
#                line=dict(color=colors[1])
#                ),
#        #color="#ff7f0e",
#        #secondary_y=False,
#        )
#
#
#        fig.add_trace(
#            go.Scatter(
#                #x =X,
#                #y =df_plot[[observationer[2], 'City']][(df_plot['City'] == city) & (df_plot[observationer[2]] > 0) & (df_plot[observationer[2]] < 45)][observationer[2]].rolling(rolling_mean).mean(),
#                y = df_plot[df_plot[tid].between(str(year)+'-01-01',str(year)+'-12-31')][observationer[2]],
#                name=year+' '+observationer[2],
#                yaxis='y3',
#                opacity=0.2,
#                line=dict(color=colors[2])
#                ),
#            #secondary_y=True,
#            #color="#d62728",
#        )
#        #fig.add_trace(
#        #    go.Scatter(
#        #        x = X,
#        #        y = df_daily[['Produktion total', 'City']][(df_daily['City'] == city)]['Produktion total'].rolling(5).mean(),
#        #        name='Produktion total 2'),
#        #    secondary_y=True,
#        #)
#
#    fig.update_layout(title = city,
#                    xaxis_showgrid=False,
#                    yaxis_showgrid=False,
#
#
#        xaxis=dict(
#            domain=[0.1, 0.9]
#        ),
#
#        yaxis=dict(
#            title="Daglig produktion total [kWh]",
#            titlefont=dict(
#                color=colors[0]
#            ),
#            tickfont=dict(
#                color=colors[0]
#            )
#        ),
#        yaxis2=dict(
#            title="Brændolie forbrug [l]",
#            titlefont=dict(
#                color=colors[1]
#            ),
#            tickfont=dict(
#                color=colors[1]
#            ),
#            anchor="free",
#            overlaying="y",
#            side="left",
#            position=0
#        ),
#        yaxis3=dict(
#            title="Elvirkningsgrad [%]",
#            titlefont=dict(
#                color=colors[2]
#            ),
#            tickfont=dict(
#                color=colors[2]
#            ),
#            anchor="x",
#            overlaying="y",
#            side="right"
#        ),
#    )
#
#    fig.update_traces(patch={"opacity":1},
#                       selector={"name":year+' '+observationer[0]}),
#    fig.update_traces(patch={"opacity":1},
#                       selector={"name":year+' '+observationer[1]}),
#    fig.update_traces(patch={"opacity":1},
#                       selector={"name":year+' '+observationer[2]})
#    return fig
#
#
#@app.callback(
#    Output(component_id='grafen_2', component_property='figure'),
#    Input(component_id='input_city', component_property='value'),
#    Input(component_id='input_year', component_property='value')
#)
#def update_output_div(input_city, input_year):
#
#    if input_city:
#        city=input_city[-1]
#        print(city)
#    else:
#        city = '013 Narsamijit'
#
#    df_plot = df[df['City'] == city][observationer + [tid, 'City']]
#    df_plot['year'] = df_plot[tid].dt.year
#    df_plot['month'] = df_plot[tid].dt.quarter
#    df_plot = df_plot.sort_values(by=['month', 'year'])
#    df_plot['month'] = df_plot['month'].astype(str)
#    df_plot['year'] = df_plot['year'].astype(str)
#
#    rolling_mean = 14
#
#
#    fig = px.violin(df_plot, x='year', y='Gram pr. kWh', color='month')
#
#    return fig

""" Run application  """
#app.run_server(host='0.0.0.0', debug=True, use_reloader=False, port=9090)

