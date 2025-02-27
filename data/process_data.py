import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath) #'messages.csv'
    categories = pd.read_csv(categories_filepath) #'categories.csv')
    df = messages.merge(categories,on='id')
    df.head()
    return df 

def clean_data(df):
    categories=df['categories'].str.split(';',expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.str.split('-').apply(lambda x:x[0]).values
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    df=df[df['related']!=0]
    df=df.drop_duplicates()
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename) #'sqlite:///disaster_df.db')
    df.to_sql(database_filename[:-3], engine, index=False)

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