import pandas as pd
import numpy as np
import pickle

def missing_values_table(df):
    """
    Input: A data frame
    Ouput: Amount of missing values,
         Percentage of missing values in each feature
    """
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns={
        0: 'Missing Values',
        1: '% of Total Values'
    })

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

def save_pickle(filename, model):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_days_in_month(d):
    """
    Function to categorize days in month
    categorize days to three period 1-10, 11-20, and 20-31
    """
    if d <11: 
        return 'first_10'
    elif 11<=d<=20: 
        return '10_to_20'
    else:
        return 'after 20'

def process_datetime(df, feature):
    df[feature+'_year'] = pd.to_datetime(df[feature]).dt.year
    df[feature+'_month'] = pd.to_datetime(df[feature]).dt.month
    df[feature+'_dow'] = pd.to_datetime(df[feature]).dt.weekday
    df[feature+'_quarter'] = pd.to_datetime(df[feature]).dt.quarter
    df[feature+'_isweeknd'] = [0 if int(d)<5 else 1 for d in df[feature+'_dow']]
    df[feature+'_month_interval']=[get_days_in_month(d.day) for d in pd.to_datetime(df[feature])]
    return df