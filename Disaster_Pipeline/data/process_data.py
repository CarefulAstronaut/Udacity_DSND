import sys
import pandas as pd
import sqlalchemy as sqla


def load_data(messages_filepath, categories_filepath):
    ''' Loads datasets into the app.
    
   INPUTS:
   messages_filepath = filepath for the .csv file of messages
   categories_filepath = filepath for the .csv file of categories
   
   OUTPUTS:
   df = DataFrame of the merged .csv files, based on the Message IDs
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, left_on='id', right_on='id' )
    return df


def clean_data(df):
    ''' Cleans datasets by splitting categories into binary claissfication columns, 
    transforms the df DataFrame, and drops duplicate messages. 
    
    INPUT: 
    df = DataFrame as output from above
    
    OUTPUT:
    df = cleansed DataFrame
    '''    
    categories = df.categories.str.split(";", expand=True)
    categories.columns = categories.iloc[0].str.split('-').str[0]
    
    for column in (categories):
        categories[column] = categories[column].str.split('-').str[1].astype(str)
        categories[column] = pd.to_numeric(categories[column], errors='coerce')
    return categories

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    df.drop_duplicates(keep='first', inplace=True)
    return df

def save_data(df, database_filepath):
    ''' Saves the dataframe that has been cleasned above into a SQLlite database 
    for the model to access in the next script.
    
    INPUT:
    df = DataFrame from above
    database_filename = database pathway to save for later access
    
    OUTPUT:
    SQLlite database at the input address
    '''
    engine = sqla.create_engine('sqlite:///data/DisasterResponse.db')
    df.to_sql(database_filepath, engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
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
