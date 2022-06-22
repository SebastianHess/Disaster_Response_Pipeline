# Importing python packages
import sys
import pandas as pd
from sqlalchemy import create_engine

# Function: Loading data 
def load_data(messages_filepath, categories_filepath):
    """
    Function:   Reading in messages and categories datasets and 
                returning merged dataset df.
    Args:       messages_filepath - Messages csv-file path, 
                categories_filepath - Categories csv-file path
    Return:     df - Merged dataset 
    """

    # Reading in files
    messages = pd.read_csv(messages_filepath, dtype=str)
    categories = pd.read_csv(categories_filepath, dtype=str)
    df = pd.merge(messages,categories, how='outer', on='id')
    return df


# Function: Cleaning data
def clean_data(df):
    """
    Function:   Cleaning merged dataset df.
    Args:       df - Merged dataset
    Return:     df - Merged and cleaned dataset
    """
    
    # Creating a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # Selecting the first row of the categories dataframe
    row = categories.iloc[0]

    # Using the selcted row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x:x[:-2]).tolist()

    # Renaming the columns of `categories`
    categories.columns = category_colnames

    # Converting category values to just numbers 0 or 1
    for column in categories:
        # Setting each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # Converting column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Dropping the original categories column from `df`
    df.drop('categories', inplace=True, axis=1)

    # Concatenating the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Dropping duplicates
    df.drop_duplicates(inplace=True)

    return df


# Function: Saving to database
def save_database(df, database_filename):
    """
    Function:   Saving merged and cleaned dataset df to a sqlite database.
    Args:       df - merged and cleaned dataset, 
                database_filename - File name of database
    Return:     disaster_response_pipeline.db - Saved sqlite database
    """

    # Saving the clean dataset into an sqlite database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response_pipeline', engine, index=False)

  
# Function: main (already provided by Udacity)  
def main():
    """
    Function:   Main function to process data and 
                saving a cleaned dataset as SQLite database.
    Args:       -
    Return:     -
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_database(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'disaster_response_pipeline.db')


if __name__ == '__main__':
    main()