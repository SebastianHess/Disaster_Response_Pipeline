# Importing python packages
import sys
import pandas as pd
from sqlalchemy import create_engine

# Function: Loading data 
def load_data(messages_filepath, categories_filepath):
    """
    load_data:   Reading in messages and categories datasets and 
                 returning merged dataset df
    Input:       messages_filepath - Messages csv-file path, 
                 categories_filepath - Categories csv-file path
    Returns:     df - Merged dataset 
    """

    # Reading in files
    messages = pd.read_csv(messages_filepath, dtype=str)
    categories = pd.read_csv(categories_filepath, dtype=str)
    df = pd.merge(messages,categories, how='outer', on='id')
    return df


# Function: Cleaning data
def clean_data(df):
    """
    clean_data:   Cleaning merged dataset df
    Input:        df - Merged dataset
    Returns:      df - Merged and cleaned dataset
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
    for col in categories:
        # Setting each value to be the last character of the string
        categories[col] = categories[col].str[-1:]

        # Converting column from string to numeric
        categories[col] = pd.to_numeric(categories[col])

    # Replacing all values of "2" in categories dataframe column "related"
    categories['related'].replace(inplace=True, to_replace=2, value=1)

    # Dropping the original categories column from `df`
    df.drop('categories', inplace=True, axis=1)

    # Concatenating the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Dropping duplicates
    df.drop_duplicates(inplace=True)

    # Filling NaN values with value "0"
    #pd.DataFrame(df).fillna(0,inplace=True)

    # Dropping column "original" because of NaN values
    df = df.drop(['original'], axis=1)

    return df


# Function: Saving to database
def save_database(df, database_filename):
    """
    save_database:   Saving merged and cleaned dataset df to a sqlite database
    Input:           df - merged and cleaned dataset,
                     database_filename - File name of database
    Returns:         disaster_response_pipeline.db - Saved sqlite database
    """

    # Saving the clean dataset into an sqlite database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response_pipeline', engine, index=False, if_exists='replace')

  
# Function: main (already provided by Udacity)  
def main():
    """
    main:      Main function to process data and 
               saving a cleaned dataset as SQLite database
    Input:     -
    Returns:   -
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