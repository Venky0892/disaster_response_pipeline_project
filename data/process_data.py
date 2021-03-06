import sys
import pandas as pd 
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Input : File path of two csv 
        messages_filepath - Message file path
        categories_filepath - Category dataset file path
    
    Output = Merged CSV
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df

def clean_data(df):
    """
    Split the values in the categories column on the character so that each value becomes a separate column.
    Input : Dataframe
    Output: Cleaned Dataframe
    """
    #Spliting categories to seperate columns
    category = df.categories.str.split(';', expand = True)

    #Getting Category name
    category_columns = category.iloc[0].apply(lambda x: x[:-2])

    category.columns = category_columns

    #Convert string to numeric 
    for col in category:
        # set each value to be the last character of the string
        category[col] = category[col].astype(str).str[-1]
        # convert column from string to numeric
        category[col] = category[col].astype(int)

    #Drop category column & Concat with new categories
    df.drop(columns = ['categories'], inplace=True,)

    # Drop the duplicates
    df = df.join(category)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Saving dataframe to database
    Input: 
        df - dataframe
        database_filename - database for cleaned dataframe
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages', engine, index=False, if_exists="replace")  



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