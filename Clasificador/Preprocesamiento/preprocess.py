import stanza
import os
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.INFO)
from helpers import *


def pipeline_registry(func):
    def wrapper(*args):
        logging.info('running function : {}'.format(func.__name__))

        return func(*args)

    return wrapper



def __pipeline__(fullpath,output_path):

    """  
        1) Lectura de archivo sin procesar
        2) Concatenación de mensajes
        3) tokenización y remoción de puntuación
        4) ONE-HOT-ENCODING

    """

    tags = read_tags()
    df = read_file(fullpath, tags)

    df = join_conversation(df)
    df = add_lemmatization(df)

    nlp = stanza.Pipeline(lang= 'es', dir= nb_path, processors='tokenize, pos')
    df['pos_tagger'] = df.apply(lambda row: add_upos(nlp,row['text']), axis=1)

    df = export_data(df, output_path)

                       
    return df



@pipeline_registry
def read_file(fullpath, tags):

    id_tagged =  tags['Id'].to_list()

    INPUT_COLS = ['Identifier','message_created_at','person_type','content2']
    df = pd.read_csv(fullpath, 
                 sep="\t",
                 encoding='utf-8',
                 usecols= INPUT_COLS
                 )
    
    df = df[df['Identifier'].isin(id_tagged)]
    
    df = df[df["person_type"] == "Client"]

    df = pd.merge(df, 
                  tags, 
                  left_on= 'Identifier',
                  right_on = 'Id',
                  how = 'inner'
                )


    return df



@pipeline_registry
def read_tags():
    path = 'drive/MyDrive/MSDS/Tesis/files/conversaciones_taggeadas_1000.csv'
    tags = pd.read_csv(path, sep = "\t", usecols= range(0,4))

    return tags



@pipeline_registry
def create_tag_cols(df):
    """ ONE HOT ENCODING """
    df["tags2"] = df['tags'].apply(convert_string_to_list)
    df = df.drop('tags2', 1).join(df.tags2.str.join('|').str.get_dummies())
    df = df.reset_index()

    df = df[df["person_type"] == "Client"]

    return df



@pipeline_registry
def join_conversation(df):
    
    """
        1) Se concatenan los mensajes asociados a una conversación.
        2) Se tokeniza y se remueve la puntuación


    """

    ids = df['Identifier'].unique()
    chunks = []
    for i in ids:

        chunk = df[df['Identifier'] == i]
        chunk['content2'].replace(to_replace=[None], value=' ', inplace=True)
        chunk['content2'] = chunk['content2'] + ";"
        msgs = ' '.join(chunk['content2'])

        process_msgs = remove_punctuation(msgs)
        process_msgs = tokenize(process_msgs)
        
        process_msgs = ' '.join(process_msgs)
        chunk['text'] = process_msgs
        chunk['text2'] = msgs
        chunk = chunk.drop_duplicates(subset=['Identifier'], keep='first')
        chunks.append(chunk)

    df = pd.concat(chunks,ignore_index=True) 
    
    
    return df



    
@pipeline_registry
def add_lemmatization(df):  

    def spacy_lemmatizer(text):
        doc = nlp(text)
        lemas = [token.lemma_ for token in doc]

        return (" ".join(lemas)) 

    nlp= es_core_news_sm.load()
    df['text'] = df['text'].apply(spacy_lemmatizer)
    df['text'] =  df['text'].apply(add_custom_lemmas)


    return df


def add_upos(nlp, text):
    doc = nlp(text) 
    upos = []
    for s in doc.sentences:
        for w in s.words:
            upos.append(w.upos)
            
    upos = ' '.join(upos)
    return upos




@pipeline_registry
def export_data(df, output_path):

    OUTPUT_COLS = list(range(0,4)) + list(range(5,10))

    df = df.iloc[:,OUTPUT_COLS]
    df.to_csv(output_path,
              sep=";",
              encoding='utf-8',
              index=False)



path = '../files'
filename = 'muestra_anonimizada5.csv'
fullpath = os.path.join(path, filename)
output_path = f'{path}/data_preprocesada.csv'

df = __pipeline__(fullpath,output_path)
print("\n\t PROCESO FINALIZADO.")