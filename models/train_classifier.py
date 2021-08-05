import sys
import pandas as pd
import sqlite3
import numpy as np
from sqlalchemy import create_engine
from string import punctuation

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.ensemble import AdaBoostClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import nltk
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    Loading a database file and converting them to dataframe
    Input:
        database_filepath - Database file path 
    Output:
        X - Message list
        Y - Target variable
    """
    # Load the database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('messages', engine)

    # define features and target
    X = df.message
    y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    
    return X, y, category_names

def tokenize(text):
    """
    Tokenzing, lemmatize and normalizing the text
    Input: 
        text - Text data
        Output - Clean Tokens
    """

    # First removing Punctutions
    exclude = set(punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)

    # Text tokenizing
    tokens = word_tokenize(text)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
   Machine learning pipleine using adaboost classifier
    Input:
       None
    Output: 
        clf: gridSearch Model
    """
    pipeline =  Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier((AdaBoostClassifier())))
    ])
    # grid search parameters
    parameters = {
    'tfidf__norm':['l2','l1'],
    'vect__stop_words': ['english',None],
    'clf__estimator__learning_rate' :[0.1, 0.5, 1, 2],
    'clf__estimator__n_estimators' : [50, 60, 70],
    }
    #create grid search object
    clf_grid_model = GridSearchCV(pipeline, parameters)

    return clf_grid_model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Classification report 
    Input:
        model -  trained model
        X_test - test data for the predication 
        Y_test -  test labels 
    Output:
        None 
    """
    # predict 
    y_pred = model.predict(X_test)
    # print the metrics
    for i, col in enumerate(category_names):
        print('{} category metrics: '.format(col))
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    """
    Exporting a model as a pickle file
    Input:
        model: trained model 
        model_filepath: location to store the model
    Output: None
    """

    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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