import os
import pickle
import pandas as pd
from typing import Tuple

from config.config import DataLoader, Variables


def prepare_customer_df(df_customers):
    # Replace the word None with NA
    df_customers['fashion_news_frequency'] = df_customers['fashion_news_frequency'].replace({'None': None})

    # Impude the mssing ages with the mean age and make the age variable to an age range
    df_customers['age'] = df_customers['age'].mean()
    df_customers['age_bins'] = pd.cut(df_customers['age'], bins=Variables.AGE_BINS, labels=Variables.AGE_LABELS)


def splitting_data(df):
    df_train = df[(df['t_dat'] >= DataLoader.TRAIN_START_DATE) & (df['t_dat'] <= DataLoader.TEST_START_DATE)]
    df_test = df[df['t_dat'] >= DataLoader.TEST_START_DATE]
    
    return df_train, df_test


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Define file paths
    articles_path = './data/parquet/df_articles.parquet'
    customers_path = './data/parquet/df_customers.parquet'
    transactions_path = './data/parquet/df_transactions.parquet'
    
    pickeled_articles_path = './data/pickeled/df_train.p'
    pickeled_train_path = './data/pickeled/df_test.p'
    pickeled_test_path = './data/pickeled/df_articles.p'
    
    # Check if the files have already been pickeled, to speed up time
    if os.path.exists(pickeled_train_path) and os.path.exists(pickeled_test_path) and os.path.exists(pickeled_articles_path):
        print('Data was already pickeled!')
        df_train = pickle.load(open(pickeled_train_path, 'rb'))
        df_test = pickle.load(open(pickeled_test_path, 'rb'))
        df_articles = pickle.load(open(pickeled_articles_path, 'rb'))
        
        return df_train, df_test, df_articles
    else:
        if not os.path.exists(articles_path) and not os.path.exists(customers_path) and not os.path.exists(transactions_path):
            # Load data
            print('Importing the data from CSV files')
            df_articles = pd.read_csv('./data/articles.csv')
            df_customers = pd.read_csv('./data/customers.csv')
            df_transactions = pd.read_csv('./data/transactions_train.csv')

            # Converting to parquet
            print('Converting to parquet file for more efficient data loading')
            df_articles.to_parquet(articles_path)
            df_customers.to_parquet(customers_path)
            df_transactions.to_parquet(transactions_path)
        
    # Load data from parquet file
    print('Importing the data from parquet files')
    df_articles = pd.read_parquet(articles_path)
    df_customers = pd.read_parquet(customers_path)
    df_transactions = pd.read_parquet(transactions_path)
    
    # Prepare customer data set
    print('Preparing customer df')
    prepare_customer_df(df_customers)
    
    # Merge data
    print('Merging the data')
    df_customers_reduced = df_customers[Variables.CUSTOMER_VARIABLES]
    df_articles_reduced = df_articles[Variables.ARTICLE_VARIABLES]
    
    df_transactions = df_transactions.merge(df_customers_reduced, on='customer_id')
    df_transactions = df_transactions.merge(df_articles_reduced, on='article_id')

    # Split the data into train and test split
    print('Splitting the data')
    df_train, df_test = splitting_data(df_transactions)
    
    # Save data sets in case of loading again
    print('Pickel the data')
    os.makedirs('./data/pickeled', exist_ok=True)  # Create directory if it doesn't exist
    pickle.dump(df_train, open('./data/pickeled/df_train.p', 'wb'))
    pickle.dump(df_test, open('./data/pickeled/df_test.p', 'wb'))
    pickle.dump(df_articles, open('./data/pickeled/df_articles.p', 'wb'))

    return df_train, df_test, df_articles