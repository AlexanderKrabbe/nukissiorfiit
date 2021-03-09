import pandas as pd
import time
from pandas_profiling import ProfileReport


colors = ["#1E88E5","#FFC107","#004D40","#D81B60"]
st = time.time()



""" READ IN DATA """
df_read = pd.read_csv('data_with_pop.csv', sep=',', dtype={'City': str}, index_col=0, parse_dates=['Date', 'Periode'])
df = df_read
df['B'] = df[['Brændolie forbrug','Forbrug brændolie']].sum(axis=1)
df['Brændolie forbrug'] = df['B']
df['Forbrug brændolie'] = df['B']


""" MAKE DAILY DATAFRAME """
df_daily = df.drop('Periode', axis = 1)
df_daily = df_daily[df_daily['Date'].notnull()]
#df_daily = df_daily.sort_values(by=['City'])


""" MAKE MONTHLY DATAFRAME """
df_monthly = df.drop('Date', axis = 1)
df_monthly = df_monthly[df_monthly['Periode'].notnull()]
#df_monthly = df_monthly.sort_values(by=['City'])

""" LOCATION VARIABLE """
tid = 'Date' #'Periode' #


#df = df.drop(['B', 'Unnamed: 0.1'], axis=1)
df = df_monthly#.dropna(axis=1, how='all')
df_temp = df
df_temp = df_temp[(df_temp['Elvirkningsgrad'] > 0) & (df_temp['Elvirkningsgrad'] < 65)]
df_temp = df_temp[(df_temp['Gram pr. kWh'] > 0) & (df_temp['Gram pr. kWh'] < 800)]
#df_temp = df_temp[(df_temp['Brændolie forbrug'] > 0) & (df_temp['Brændolie forbrug'] < 2500)]
#df_temp = df_temp[(df_temp['Produktion til byen'] > 0) & (df_temp['Produktion til byen'] < 14000)]
#df_temp = df_temp[(df_temp['Produktion total'] > 0) & (df_temp['Produktion total'] < 8000)]
#df_temp = df_temp[(df_temp['Vejlys'] > 0) & (df_temp['Vejlys'] < 580)]
#df_temp = df_temp[(df_temp['Eget forbrug'] > 0) & (df_temp['Eget forbrug'] < 1400)]


df_report = df_temp#[['Brændolie forbrug',
       #'Produktion til byen',
       #'Gram pr. kWh', 'Elvirkningsgrad',
       #'Produktion total', 'Vejlys', 'Eget forbrug', 'Date', 'City',
       #'City_code', 'Distrikt', 'pop']]


profile = ProfileReport(df_report, minimal=True)
profile.to_file(output_file="output_min_monthly.html")

prof = ProfileReport(df_report) 
prof.to_file(output_file='output_monthly.html')



