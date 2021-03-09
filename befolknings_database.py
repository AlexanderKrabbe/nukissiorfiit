import pandas as pd
import datetime as datetime
import time

st =time.time()

df_befolkning = pd.read_csv('befolkning.csv', sep=';',skiprows=[0,1])#, parse_dates=['tid'])
df_befolkning = df_befolkning.set_index('tid')
befolkning = df_befolkning.to_dict()

df_read = pd.read_csv('panda_with_districts.csv', sep=',', dtype={'City': str}, parse_dates=['Date', 'Periode'])#,index_col=0, )
#df_read.sort_values(by=['Date','City'], inplace=True)
df_read['pop'] = -500
stamdata = {}

for i,k in enumerate(befolkning):
    #print(i,k)    
    new_key = k.split(" ")
    #print(len(new_key))
    if new_key[1] == 'Kujalleq' or new_key[1] == 'Paa':
        new = new_key[0] + ' ' + new_key[1]
        #print(new)
    else:
        new = new_key[0]
    stamdata[new] = befolkning[k]
#    print(new_key)

#print(stamdata)
#print(df_read['City'].unique())
x=0
liste_over_afviste_byer = []
for i in df_read.index.to_list():
    #print(i)
    x+=1
    y = df_read['Date'].iloc[i].year
    if pd.isnull(y):
        y = df_read['Periode'].iloc[i].year
        #print(y)

    by = df_read['City'].iloc[i]

    try:
        pop = stamdata[by[4:]][int(y)]
    except ValueError as e:
        #print(e)
        print(by, y, i)
        pop = -500
        #print(df_read.iloc[i])
        #break

    df_read.loc[i,'pop'] = pop
    #if by == '021 Saarloq':
     #   print(pop)
      #  print(df_read[['City','Date','Periode', 'pop']].iloc[i])



    #if x % 5000 == 0:
    #if  df_read['pop'].iloc[i] == -500:
        #print(pop)    
        #print(df_read['pop'].iloc[i])
        #print(by[4:],y,pop)
        #print(df_read[['City','Date','Periode', 'pop']].iloc[i])
        #break

print(time.time()-st)
df_read.to_csv('data_with_pop.csv')
#print(df_read[df_read['pop'] == -500]['City'].unique())
