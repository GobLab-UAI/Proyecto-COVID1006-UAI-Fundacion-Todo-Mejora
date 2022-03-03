import os
import pandas as pd
import numpy as np
from datetime import datetime


path = '../files'
filename = 'muestra_anonimizada5.csv'
fullpath = os.path.join(path, filename)



df = pd.read_csv(fullpath, 
             sep="\t",
             encoding='utf-8'
             )


def get_conversation_duration(start,end):    
    fmt = '%Y-%m-%d %H:%M:%S'
    start = datetime.strptime(start, fmt)
    end = datetime.strptime(end, fmt)
    
    minutes_diff = round((end - start).total_seconds() / 60.0,0)
    
    return minutes_diff

   
def get_status_considerar(user_time_response):
    if user_time_response > 0:
        return "Y"
    else:
        return "N"


def get_text_length(col):

    col = ' '.join(col['content2'])
    col = col.split(' ')

    return len(col)


identifier = df['Identifier'].unique()

result = {
    'Identifier':[],
    'start_conversation':[],
    'end_conversation':[],
    'conversation_duration':[],
    'user_entries':[],
    'client_entries':[],
    'client_lenght':[],
    'client_lenght_avg':[],
    'client_msg_freq':[],
    'rate_user_client':[],
    'user_response_avg':[],
    'client_response_avg':[]
    }

for i in identifier:
    filtered = df[df['Identifier']== i]
    filtered = filtered.reset_index()
    start_conversation = filtered['message_created_at'].iloc[0]
    end_conversation = filtered['message_created_at'].iloc[-1]
    conversation_duration = get_conversation_duration(start_conversation,end_conversation)
    user_entries = len(filtered[filtered['person_type'] == 'User'])
    client_entries = len(filtered[filtered['person_type'] == 'Client'])
    try:
        client_lenght =  get_text_length(filtered[filtered['person_type'] == 'Client'])
    except:
        client_lenght =  1
    try:
        client_lenght_avg = round(client_lenght / client_entries,2)
    except:
        client_lenght_avg = 0
    try:
        rate_user_client = round(user_entries/client_entries,2)
    except:
        rate_user_client = 0
    user_response_avg = []
    client_response_avg = []
    client_response_avg = []
    client_msg_freq = []
    
    
    
    
    for index, row in filtered.iterrows():
        if index > 0:
            
            if row['person_type'] == "Client" and filtered['person_type'].iloc[index-1]== "Client":
                previous_msg = filtered['message_created_at'].iloc[index-1]
                current_msg = row['message_created_at']
                msg_duration = get_conversation_duration(previous_msg, current_msg)
                client_msg_freq.append(msg_duration)

            elif row['person_type'] == "Client" and filtered['person_type'].iloc[index-1]== "User":
                user_request = filtered['message_created_at'].iloc[index-1]
                client_response = row['message_created_at']
                client_response_time = get_conversation_duration(user_request,client_response)
                client_response_avg.append(client_response_time)
            elif row['person_type'] == "User" and filtered['person_type'].iloc[index-1]== "Client":
                client_request = filtered['message_created_at'].iloc[index-1]
                user_response = row['message_created_at']
                user_response_time = get_conversation_duration(client_request,user_response)
                user_response_avg.append(user_response_time)

    if len(user_response_avg) > 0:
        user_response_avg = round(pd.Series(user_response_avg).mean(),2)
    else:
        user_response_avg = -1

    if len(client_msg_freq) > 0:
        client_msg_freq = round(pd.Series(client_msg_freq).mean(),2)
    else:
        client_msg_freq = 0
    
    if len(client_response_avg) > 0:
        client_response_avg = round(pd.Series(client_response_avg).mean(),2)
    else:
        client_response_avg = -1
        
    result['Identifier'].append(i)
    result['start_conversation'].append(start_conversation)
    result['end_conversation'].append(end_conversation)
    result['conversation_duration'].append(conversation_duration)
    result['user_entries'].append(user_entries)
    result['client_entries'].append(client_entries)
    result['client_lenght'].append(client_lenght)
    result['client_lenght_avg'].append(client_lenght_avg)
    result['rate_user_client'].append(rate_user_client)
    result['user_response_avg'].append(user_response_avg)
    result['client_response_avg'].append(client_response_avg)
    result['client_msg_freq'].append(client_msg_freq)
    

   
final_df = pd.DataFrame(result)


filter_df = final_df[final_df['user_response_avg']>-1]
filter_df = filter_df[filter_df['client_response_avg']>-1]
filter_df.to_csv('tiempos_de_respuesta.csv',sep=";",encoding='utf-8',index=False)

print("Proceso finalizado")