import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads messages and categories datasets and merges them into a single dataframe.

    INPUT
        messages_filepath - string for filepath of the messages dataset
        categories_filepath - string for filepath of the categories dataset

    OUTPUT
        a dataframe of merged messages and categories datasets
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    '''
    Cleans dataframe.
    
    INPUT
        df - dataframe to clean

    OUTPUT
        a dataframe with binary categorical values and without duplicates     
    '''

    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # Extract a list of new column names for categories.
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])

    # Rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    # Drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filename):
    '''
    Saves a dataframe into a table named `messages` in the sql database provided and
    each time this function runs, it replaces the table if exists in the database.

    INPUT
        df - dataframe to save
        database_filename - string for the filepath of the database    
    '''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, messages_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()