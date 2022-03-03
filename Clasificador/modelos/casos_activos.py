import pandas as pd
from datetime import datetime


def string_to_date(date_str1):
    date_dt1 = datetime.strptime(date_str1, '%Y-%m-%d')

    return date_dt1


def datetime_from_string(string):
    string = string[0:10]
    date = dt.strptime(string, '%Y-%m-%d')
        
    return date


def create_datecols(date):
    dt.strptime('2019-10-18', '%Y-%m-%d')
    if date> dt.strptime('2019-10-17', '%Y-%m-%d') and date <dt.strptime('2019-12-31', '%Y-%m-%d'):
        return 'estallido'
    elif date> dt.strptime('2020-05-20', '%Y-%m-%d') and date <dt.strptime('2020-07-01', '%Y-%m-%d'):
        return 'pandemia'
    elif date> dt.strptime('2021-03-11', '%Y-%m-%d') and date <dt.strptime('2021-06-26', '%Y-%m-%d'):
        return 'pandemia'
    else:
        return ''


# casos activos obtenidos del minciencia
fullpath = 'https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto5/TotalesNacionales.csv'

data = pd.read_csv(fullpath)
data = data.iloc[[4],:]
data = pd.melt(data, value_vars= data.iloc[:,1:].columns,value_name= 'activos', var_name= 'fechas')
data['fechas'] = data['fechas'].apply(string_to_date, 1)


# Visualización de la curva de casos activos para determinar fechas críticas

import plotly.express as px
fig = px.line(data, x='fechas', y="activos")
fig.show()


df = pd.read_csv('../files/data_preprocesada.csv',
                                     sep = ";"
                                    )

df['fecha'] = df['message_created_at'].apply(datetime_from_string)
df['flag'] = df['fecha'].apply(create_datecols, 1)
agg = df.groupby(['flag']).count()


tags = pd.get_dummies(df['flag'])
df2 = pd.concat([df, tags], 1)
df2 = df2.iloc[:,[0,-1,-2]]
df2.to_csv('../files/pandemia_estallido.csv', sep=";", index = False)

