# Importing python packages
import sys
import nltk
nltk.download(['stopwords', 'punkt', 'wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

# Function: Loading data from database
def load_data(database_filepath):
    """
    load_data:   Loading the data from database and defining.
    			 X an y variables
    Input:       disaster_response_pipeline.db - database,
    			 df - data frame              
    Returns:     X - Explanatory variable,
    			 y - Response variable,
    			 category_names - Label names for visualization
    """
    # Loading file .db
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response_pipeline', con=engine)

    # Defining X and y variables
    X = df['message']
    y = df.drop(['id', 'message', 'genre'], axis=1)
    category_names = y.columns

    return X, y, category_names

# Function: Tokenization function to process text data
def tokenize(text):
    """
    tokenize:   Tokenization function to process text data
    Input:      text - as string
    Returns:    clean_tokens
    """
    # tokenize text 
    tokens = word_tokenize(text)
    
    # remove stop words
    stop_words = stopwords.words("english")
    tokens = [tok for tok in tokens if tok not in stop_words]
    
    # lemmatize tokens
    lemmatizer = nltk.WordNetLemmatizer()

    # create clean tokens
    clean_tok = []
    for tok in tokens:
        clean_tok_1 = lemmatizer.lemmatize(tok).lower().strip()
        clean_tok.append(clean_tok_1)

    return clean_tok

# Function: Building ML model
def build_model():
    """
    build_model:   Bulding a gridsearch object as final model pipeline
    Input:         pipeline - ML model pipeline,
                   parameters - GridSearchCV parameters
    Returns:       cv - Final model pipeline
    """

    # Modelling ML pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(criterion='gini', n_estimators=50, max_depth=2, random_state=1)))
        ])

    # Defining parameters for GridSearchCV
    parameters = {
            #'vect__ngram_range': ((1, 1), (1, 2)),
            #'clf__estimator__min_samples_split': [2, 2],
            'clf__estimator__n_estimators': [100],
            'clf__estimator__max_depth': [5, 50]
            }

    # Creating gridsearch object and return as final model pipeline
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)
    
    return cv

# Function: Evaluating ML model
def evaluate_model(model, X_test, y_test, category_names):
    """
    evaluate_model: Creating ML model
    Inputs:         model - ML model, 
                    X_test - Explanatory test variable,
                    y_test - Response test variable,
                    category_names - Label names for visualization
    Returns:        ml_model - Print of visualization of ML model
    """    
    # Defining predict test data
    y_pred = pd.DataFrame(model.predict(X_test))
    
    # Getting number or columns of y
    columns_count = len(y.columns)

    # For loop to creating f1 score precision
    for i in range(columns_count):
        print(' ==================================================== ', '\n',
              'Report of category: ' + ' *** ' + y.columns[i] + ' *** ', '\n',
              '==================================================== ', '\n',
              classification_report(y_test.iloc[i],y_pred.iloc[i]),
              '==================================================== ', '\n\n')

# Function: Saving ML model as pickle file
def save_model(model, model_filepath):
    """
    save_model:   Exporting model to pickle file.
    Input:        model - ML model
    Returns:      model_filepath - Pickle file path
    """

    # Exporting model to pickle file
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    """
    main:      Main function to train ML model.
    Input:     -
    Returns:   Information about:
                - Building model
                - Training model
                - 
    """
if len(sys.argv) == 3:
    database_filepath, model_filepath = sys.argv[1:]
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, y, category_names = load_data(database_filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, y_test, category_names)

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