import numpy as np
import pandas as pd 
import json
import pickle
from joblib import dump, load
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score,confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)
import time


def registry(fn):
    def wrapper(*args, **kwargs):
        logging.info('Running function: {}'.format(fn.__name__))
        return fn(*args, **kwargs)
    return wrapper



class Model():  
    def __init__(self, data_x, data_y, k, model, prob, tag):

        """ 
        @params:
            data_x : Dataframe con las columnas id y la conversación.
            data_y: Tag
            k : Número de folds para validación cruzada
            model: Modelo ML para entrenamiento
            prob: (True = probabilidad, False: Clase)
            tag: 'Depresiva', 'Ansiosa' o 'Suicida'

        @public methods:
            feature_extraction: Genera matriz TF-IDF
            get_metrics_results: Genera Dataframe con precision y recall

        """


        self.data_x = data_x
        self.data_y = data_y
        self.k = k
        self.model = model
        self.prob = prob
        self.tag = tag


    def feature_extraction(self, col, k):

        spanish_stopwords = stopwords.words('spanish')

        X_train, X_test, y_train, y_test = train_test_split(
            self.data_x, self.data_y, test_size=0.3)
            

        tfidf = TfidfVectorizer(min_df=1, 
                                stop_words = spanish_stopwords,
                                ngram_range= (1,3))  

        X_train = tfidf.fit_transform(X_train[col])


        X_train = pd.DataFrame(X_train.toarray(),
                                columns=tfidf.get_feature_names())


        # guardando modelo
        dump(tfidf,'../files/models/tfidf_{}.joblib'.format(col)) 

        # leyendo modelo
        tfidf = load('../files/models/tfidf_{}.joblib'.format(col)) 

        X_test = tfidf.transform(X_test[col])
        X_test = pd.DataFrame(X_test.toarray(),
                                columns=tfidf.get_feature_names())




        y_pred, clf = self._fitter(X_train, y_train, X_test)
        importance = self.get_feature_importance(clf,X_train, y_train, k)


        # save to npy file
        np.save('../files/kw_extraction/{}_{}.npy'.format(col,self.tag), importance)

        return importance

    
    def get_feature_importance(self,clf, X_train, y_train, n_words):
        from sklearn.feature_selection import SelectFromModel
        feature_importance = {'words':None,'coef':None}

        feature_names = np.array(X_train.columns)
        importance = np.abs(clf.coef_)
        importance = importance[0]

        threshold = np.sort(importance)[-n_words - 1] + 0.01
        sfm = SelectFromModel(clf, threshold=threshold).fit(X_train, y_train)
        coef_positions = sfm.get_support()
        feature_importance['words'] = feature_names[coef_positions]

        i, = np.where(coef_positions == True)
        top_n = importance[i]
        feature_importance['coef'] = top_n

        importance = pd.DataFrame(feature_importance)

        importance = np.array([importance['words']])

        return importance
        

    @registry
    def _fitter(self, x_train, y_train, x_test, **kwargs):
        clf = self.model(**kwargs)
        clf.fit(x_train, y_train)

        if self.prob == True:
            y_pred = clf.predict(x_test)
            y_pred_prob = clf.predict_proba(x_test)
            return y_pred, y_pred_prob
        else:
            y_pred = clf.predict(x_test) 
            return y_pred, clf



    def get_top_positive_class(self, n, **kwargs):
        df = self.get_metrics_results(**kwargs)

        fp = df[df[self.tag] == 0]
        fp = fp[fp['y_pred'] == 1]
        fp = fp.sort_values(by='y_pred_prob', ascending=False)
        fp = fp.iloc[:5,[1,2]]
        fp['tag'] = self.tag
        fp['obs'] = 'falso positivo'


        fn = df[df[self.tag] == 1]
        fn = fn[fn['y_pred'] == 0]
        fn = fn.sort_values(by='y_pred_prob', ascending=True)
        fn = fn.iloc[:5,[1,2]]
        fn['tag'] = self.tag
        fn['obs'] = 'falso negativo'


        result = pd.concat([fp,fn],axis=0)
        
        result['y_pred_prob'] = round(df['y_pred_prob'],2)

        result.to_csv('top_misclassification_{}.csv'.format(self.tag),index= False)

        return result


    def get_metrics_results(self, **kwargs):
        chunks = []
        cross_val = self.cross_validation()
        
        for x_train, y_train, x_test, y_test in cross_val:
            y_pred, y_pred_prob = self._fitter(x_train, y_train.iloc[:,[0]], x_test, **kwargs)
            
            pred_prob = []
            for i in y_pred_prob:
                pred_prob.append(i[1])

            y_test['y_pred_prob'] = pred_prob
            y_test['y_pred'] = y_pred  
            
            chunks.append(y_test)

        df = pd.concat(chunks, ignore_index= True)
        df = df.drop_duplicates(subset=['Identifier'], keep='first')
        return df



    @registry
    def cross_validation(self):
        kf = KFold(n_splits = self.k, shuffle = True)


        for train_index, test_index in kf.split(self.data_x, self.data_y):
            x_train =  data_x.iloc[train_index]
            x_test  =  data_x.iloc[test_index]
            y_train =  pd.concat([data_y.iloc[train_index],
                                  data_x.iloc[train_index,[0]] ],axis=1)
            
            y_test =  pd.concat([data_y.iloc[test_index],
                                  data_x.iloc[test_index,[0]] ],axis=1)
            

            tfidf = TfidfVectorizer(min_df=0.01, ngram_range= (1,3))  

            x_train = tfidf.fit_transform(x_train['text'])
            x_train = pd.DataFrame(x_train.toarray(),
                                   columns=tfidf.get_feature_names())

            x_test = tfidf.transform(x_test['text'])
            x_test = pd.DataFrame(x_test.toarray(),
                                  columns=tfidf.get_feature_names())


            yield x_train, y_train, x_test, y_test

