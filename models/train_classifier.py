import sys
# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

# import libraries
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import parallel_backend


def load_data(database_filepath):
    '''
    Load data from a sqlite database, that has messages with their multi-category data.
    And return features(message texts), targets/labels(multi-category for each message)
    and list of category names.
    
        Parameters:
            database_filepath:
                string for filepath of the sqlite database. 

        Returns:
            X:
                numpy array for features(message texts).
            y:
                numpy array for targets/labels(multi-category for each message).
            categories: 
                list of strings for category names.    
    '''

    # Load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)

    # Extract feature & target values X # y and categories names
    X = df.message.values 
    y = df.iloc[:, 4:].values
    categories = df.iloc[:, 4:].columns.tolist()

    return X, y, categories


def tokenize(text):
    '''
    Use nltk to case normalize, lemmatize, and tokenize text.
    It is used in the machine learning pipeline to vectorize and then apply TF-IDF to the text. 

    Parameters:
        text: 
            string of the text needs to be tokenized.

    Returns:
        tokens:
            list of cleaned, normalized, lammatized tokens.    
    '''

    # Remove urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, 'urlplaceholder', text)
    
    # Convert to lower case and remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ' , text.lower()) 
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    
    # lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def build_model():
    '''
    Build a pipeline that processes text and then performs multi-output classification on the 36 categories in the dataset. 
    GridSearchCV is used to find the best parameters for the model.

    Returns:
        model:
            a multi-output classifier model.
    '''
    # Build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=0, n_estimators=10)))
    ])

    # Use GridSearchCV to find best parameters 
    parameters = { 'clf__estimator__criterion': ['gini', 'entropy'] }
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=3)

    return model


def display_results(y_test, y_pred, category_names):
    '''
    Print classification report for the given true and predicted labels.

    Parameters:
        y_test: 
            1d array-like, or label indicator array / sparse matrix of ground truth (correct) target values.
        y_pred : 
            1d array-like, or label indicator array / sparse matrix of estimated targets as returned by a classifier.
        category_names:
            list of strings for category names.
    '''

    print(classification_report(y_test, y_pred, target_names=category_names))


def display_results_details(y_test, y_pred, category_names):
    '''
    Print classification report for the given true and predicted labels, for the two classes of each category.

    Parameters:
        y_test: 
            1d array-like, or label indicator array / sparse matrix of ground truth (correct) target values.
        y_pred : 
            1d array-like, or label indicator array / sparse matrix of estimated targets as returned by a classifier.
        category_names:
            list of strings for category names.
    '''

    for i in range(len(category_names)):
        print ('Category: ' + category_names[i])
        print(classification_report(y_test[:, i], y_pred[:, i]))


def evaluate_model(model, X_test, y_test, category_names):
    '''
    Print classification report and accuracy score for model performance.

    Parameters:
        model:
           a multi-output classifier model. 
        X_test: 
            iterable of data(message texts) for the model to predict on.
        y_test: 
            1d array-like, or label indicator array / sparse matrix of ground truth (correct) target values.
        category_names:
            list of strings for category names.
    '''

    # predict on test data
    print('Run model predection on test data...')
    y_pred = model.predict(X_test)

    # display evaluation results
    print('Calculating classification report for model...')
    display_results(y_test, y_pred, category_names)

    # display detailed evaluation results
    print('Calculating classification report of the two classes of each category for model...')
    display_results_details(y_test, y_pred, category_names)

    # Calculate accuracy score
    print('Calculating accuracy score for model...')
    print(accuracy_score(y_test.reshape(-1, 1), y_pred.reshape(-1, 1)))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model.best_estimator_, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        with parallel_backend('multiprocessing'):
            print('Building model...')
            model = build_model()
        
            print('Training model...\n')
            model.fit(X_train, y_train)
        
            print('Evaluating model...\n')
            evaluate_model(model, X_test, y_test, category_names)

            print('Saving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)

            print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'models/train_classifier.py data/disaster_messages.db models/classifier.pkl')


if __name__ == '__main__':
    main()