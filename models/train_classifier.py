# Importing python packages
import sys
import nltk
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
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
    Function:   Loading the data from database and defining.
    			X an y variables.
    Args:       disaster_response_pipeline.db,
    			df
                
    Return:     X - Explanatory variable,
    			y - Response variable,
    			category_names - Label names for visualization
    """

    # Loading file .db
	engine = create_engine('sqlite:///' + database_filepath)
	df = pd.read_sql_table('disaster_response_pipeline', con=engine)
	
	# Defining X and y variables
	X = df['message']
	y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
	category_names = y.columns
	return X, y, category_names

# Function: Tokenization function to process text data
def tokenize(text):
    """
    Function:   Tokenization function to process text data.
    Args:       text - Text of messages
    Return:     clean_tokens - Tokenized text messages
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# Function: Building ML model
def build_model():
    """
    Function:   Bulding a gridsearch object as final model pipeline.
    Args:       pipeline - ML model pipeline,
    			parameters - GridSearchCV parameters
    Return:     cv - Final model pipeline
    """

    # Modelling ML pipeline
    pipeline = Pipeline([
    	('vect', CountVectorizer(tokenizer=tokenize)),
    	('tfidf', TfidfTransformer()),
    	('clf', MultiOutputClassifier(RandomForestClassifier(criterion='gini', n_estimators=50, max_depth=2, random_state=1)))
    	])

    # Defining parameters for GridSearchCV
    parameters = {
			'vect__ngram_range': ((1, 1), (1, 2)),
			'clf__estimator__min_samples_split': [2, 2]
			}

    # Creating gridsearch object and return as final model pipeline
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=2, n_jobs=-1)
    return cv


# Function: Evalating ML model
def evaluate_model(model, X_test, y_test, category_names):
    """
	Function:   Creating ML model.
    Args:       model - ML model, 
    			X_test - Explanatory test variable,
    			y_test - Response test variable,
    			category_names - Label names for visualization
    Return:		ml_model - Print of visualization of ML model
    """


    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Defining predict test data
    y_pred = model.predict(X_test)
    y_true = y_test.values

# Reporting f1 score precision
def ml_model(y_true, y_pred):
	"""
	Function:   Reporting f1-score precision data.
	Args:       y_true - Test values,
				y_pred - Prediction values
				Return:     Classification report for each column.
	"""
	
	# For loop to create f1 score precision
	for i in range(36):
		print('Report: ' + y_test.columns[i],'\n',
			classification_report(y_true[:, i],y_pred[:, i], output_dict=False, zero_division=1)
			)
     
    	# Using model
		ml_model(y_true, y_pred)

# Function: Saving ML model as pickle file
def save_model(model, model_filepath):
    """
	Function:   Exporting model to pickle file.
    Args:       model - ML model
    Return:  	model_filepath - Pickle file path
    """

    # Exporting model to pickle file
    pickle.dump(model,open(model_filepath,'wb'))


def main():
	"""
	Function:   Main function to train ML model.
    Args:       -
    Return:		-
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