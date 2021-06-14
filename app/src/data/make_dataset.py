import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from app import cos


def make_dataset(path, timestamp, target, cols_to_remove):

    """
        Function that creates the dataset used for training the model.

        Args:
           path (str):  path to dataset.
           timestamp (float):  time in seconds.
           target (str):  dependent variable.

        Kwargs:

        Returns:
           DataFrame, DataFrame. Train and Test datasets for the model
    """

    print('---> Getting data')
    df = get_raw_data_from_local(path)
    print('---> Train / test split')
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=24)
    print('---> Transforming data and making Feature Engineering')
    train_df, test_df = transform_data(train_df, test_df, timestamp, target, cols_to_remove)
    print('---> Preparing data for training')
    train_df, test_df = pre_train_data_prep(train_df, test_df, timestamp, target)
   
    return train_df.copy(), test_df.copy()


def get_raw_data_from_local(path):

    """
        Function to obtain de original data from local

        Args:
           path (str):  path to dataset.

        Returns:
           DataFrame. Dataset with the input data.
    """

    df = pd.read_csv(path)
    return df.copy()


def transform_data(train_df, test_df, timestamp, target, cols_to_remove):

    """
        Function that transforms the input dataset and make Feature Engineering.

        Args:
           train_df (DataFrame):  Train dataset.
           test_df (DataFrame):  Test dataset.
           timestamp (float):  Time in seconds.
           target (str):  Dependent variable.
           cols_to_remove (list): columns to drop.

        Returns:
           DataFrame, DataFrame. Train and Test datasets for the model.
    """
    # Removing senseless data related to 'impossible' beam destinations
    print('------> Removing senseless data')
    train_df = remove_senseless(train_df)
    test_df = remove_senseless(test_df)

    #Adding new predictors (Feature Engineering)
    print('------> Adding new predictors')
    train_df = add_predictors(train_df)
    test_df = add_predictors(test_df)

    #Removing rows with BM 'No beam'
    print('------> Removing data with BM=NoBeam')
    train_df = remove_rows_BM_zero(train_df)
    test_df = remove_rows_BM_zero(test_df)

    # Removing BM column
    print('------> Removing BM columns')
    train_df = remove_unwanted_columns(train_df, cols_to_remove)
    test_df = remove_unwanted_columns(test_df, cols_to_remove)

    # Saving the predictors (columns) and target in IBM COS
    print('---------> Saving predictors and target')
    cos.save_object_in_cos(train_df.columns, 'predictors_and_target', timestamp)

    return train_df.copy(), test_df.copy()


def remove_unwanted_columns(df, cols_to_remove):
    """
        Función para quitar variables innecesarias

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    return df.drop(columns=cols_to_remove)

def remove_senseless(df):
    """
        Function to remove imposible beam destinations

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    index_names_BD5 = df[ (df['BD_2'] == 1) & (df['BD_1'] == 0) & (df['BD_0'] == 1)].index
    index_names_BD6 = df[ (df['BD_2'] == 1) & (df['BD_1'] == 1) & (df['BD_0'] == 0)].index
    index_names_BD7 = df[ (df['BD_2'] == 1) & (df['BD_1'] == 1) & (df['BD_0'] == 1)].index
    df.drop(index_names_BD5, inplace = True)
    df.drop(index_names_BD6, inplace = True)
    df.drop(index_names_BD7, inplace = True)
    
    return (df)

def add_predictors(df):
    """
        Function to add new predictors (Feature Engineering)

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    df['Section_1'] = ((df['GV1'] == 1) & (df['GV2'] == 1) & (df['VBP1']==1) & (df['VBP2']==1)).astype(int)
    df['Section_2'] = ((df['GV3'] == 1) & (df['GV4'] == 1) & (df['VBP3']==1) & (df['VBP4']==1)).astype(int)
    df['Section_3'] = ((df['GV5'] == 1) & (df['VBP5']==1)).astype(int) 
    df['Section_4'] = ((df['GV6'] == 1) & (df['GV7'] == 1) & (df['VBP6']==1) & (df['VBP7']==1)).astype(int) 
    df['BtT'] = (df['Section_1'] & df['Section_2'] & df['Section_3'] & df['Section_4']).astype(int)
    return (df)

def remove_rows_BM_zero(df):
    """
        Function to remove data rows with BM zero

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    index_names = df[ (df['BM'] == 0) ].index
    df.drop(index_names, inplace = True)
    return (df)

def pre_train_data_prep(train_df, test_df, timestamp, target):
    """
        Function that makes the last transformations on the dataset before training the model
        (NULL imputing and scaling)
        Args:
           train_df (DataFrame):  Train dataset.
           test_df (DataFrame):  Test dataset.
           timestamp (float):  Time in seconds
           target (str):  Dependent variable.
        Returns:
           DataFrame, DataFrame. Train and Test datasets ready for the model.
    """

    # Split target variables before imputing and scaling
    train_target = train_df[target].copy()
    test_target = test_df[target].copy()
    train_df.drop(columns=[target], inplace=True)
    test_df.drop(columns=[target], inplace=True)

    # NULL imputing
    print('------> Inputing missing values')
    train_df, test_df = input_missing_values(train_df, test_df, timestamp)

    # Scaling
    print('------> Scaling features')
    train_df, test_df = scale_data(train_df, test_df)

    # Join the target variable to the datasets
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    train_target.reset_index(drop=True, inplace=True)
    test_target.reset_index(drop=True, inplace=True)
    train_df = train_df.join(train_target)
    test_df = test_df.join(test_target)

    return train_df.copy(), test_df.copy()

def input_missing_values(train_df, test_df, timestamp):
    """
        Function for NULLs imputing
        Args:
           train_df (DataFrame):  Train dataset.
           test_df (DataFrame):  Test dataset.
           timestamp (float):  Time in seconds.
        Returns:
           DataFrame, DataFrame. Train and Test datasets for the model.
    """
    # create an imputer that fills with 0 the potential NULLs 
    imputer = SimpleImputer(strategy='constant', fill_value=0)

    # imputing train dataset
    train_df = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)
    # imputing test dataset
    test_df = pd.DataFrame(imputer.transform(test_df), columns=test_df.columns)

    # save the imputer for future new data
    print('------> Saving imputer on the cloud')
    cos.save_object_in_cos(imputer, 'imputer', timestamp)

    return train_df.copy(), test_df.copy()

def scale_data(train_df, test_df):
    """
        Function to scale variables
        Args:
           train_df (DataFrame):  Train dataset.
           test_df (DataFrame):  Test dataset.
        Returns:
           DataFrame, DataFrame. Train and Test datasets for the model.
    """

    # objeto de escalado en el rango (0,1)
    scaler = StandardScaler()
    # scaling train dataset
    train_df = scaler.fit_transform(train_df)
    # scaling test dataset
    test_df = scaler.transform(test_df)

    return train_df.copy(), test_df.copy()