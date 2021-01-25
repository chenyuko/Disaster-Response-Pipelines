import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score 
from sklearn.base import BaseEstimator, TransformerMixin
import pickle 

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('select * from '+database_filepath.split('/')[-1][:-3] ,con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    categories_name=Y.columns
    return X,Y,categories_name

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls=re.findall(url_regex,text)
    for url in detected_urls:
        text.replace(url,'urlplaceholder')
    tokens=word_tokenize(text)
    tokens_clean=[i.lower() for i in tokens if i.lower() not in stopwords.words('english')]
    return tokens_clean


def build_model():
    pipeline_rf=Pipeline([('count_vect',CountVectorizer(tokenizer=tokenize)),('tfidf_vect',TfidfTransformer()),
          ('clf',MultiOutputClassifier(RandomForestClassifier()))])
    params_rf = {'clf__estimator__n_estimators':[50,100]}
    gs = GridSearchCV(estimator=pipeline_rf,param_grid=params_rf,cv=5)
    return gs

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self,X, y=None):
        return self
    def transform(self,X):
        return pd.Series(X).apply(lambda x:len(x)).values

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            #print(pos_tags)
            if len(pos_tags)>0:
                first_word, first_tag = pos_tags[0]
            else:
                continue
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model_with_new_feature():
    pipeline_rf=Pipeline([("Features",
                    FeatureUnion([("textpipe",Pipeline([('count_vect',
                        CountVectorizer(tokenizer=tokenize)),('tfidf_vect',TfidfTransformer())])),
                        ("textlength",TextLengthExtractor()),
                        ('starting_verb', StartingVerbExtractor())])),
                    ('clf',MultiOutputClassifier(RandomForestClassifier()))])
    params_rf = {'clf__estimator__n_estimators':[50,100]}
    gs = GridSearchCV(estimator=pipeline_rf,param_grid=params_rf,cv=5)
    return gs

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        print('Testing performance for {}: '.format(category_names[i]))
        print('  accuracy:{}, f1-score:{},precision:{},recall:{}',
              accuracy_score(np.array(Y_test)[:,i],y_pred[:,i]),
              f1_score(np.array(Y_test)[:,i],y_pred[:,i]),
              precision_score(np.array(Y_test)[:,i],y_pred[:,i]),
              recall_score(np.array(Y_test)[:,i],y_pred[:,i]))
    print('Average testing accuracy for all features: ',(np.array(Y_test)==y_pred).mean().mean())

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as handle:
        pickle.dump(model,handle)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        #gs_model = build_model()
        gs_model = build_model_with_new_feature()
        print('Training model...')
        model=gs_model.fit(X_train, Y_train)
        model = model.best_estimator_.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()