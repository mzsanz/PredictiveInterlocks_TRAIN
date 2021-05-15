import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from ..features.feature_engineering import feature_engineering
from app import cos


def make_dataset(path, timestamp, target, cols_to_remove, model_type='RandomForest'):

    """
        Función que permite crear el dataset usado para el entrenamiento
        del modelo.

        Args:
           path (str):  Ruta hacia los datos.
           timestamp (float):  Representación temporal en segundos.
           target (str):  Variable dependiente a usar.

        Kwargs:
           model_type (str): tipo de modelo usado.

        Returns:
           DataFrame, DataFrame. Datasets de train y test para el modelo.
    """

    print('---> Getting data')
    df = get_raw_data_from_local(path)
    print('---> Train / test split')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=50)
    print('---> Transforming data')
    train_df, test_df = transform_data(train_df, test_df, timestamp, target, cols_to_remove)
   
    return train_df.copy(), test_df.copy()


def get_raw_data_from_local(path):

    """
        Función para obtener los datos originales desde local

        Args:
           path (str):  Ruta hacia los datos.

        Returns:
           DataFrame. Dataset con los datos de entrada.
    """

    df = pd.read_csv(path)
    return df.copy()


def transform_data(train_df, test_df, timestamp, target, cols_to_remove):

    """
        Función que permite realizar las primeras tareas de transformación
        de los datos de entrada.

        Args:
           train_df (DataFrame):  Dataset de train.
           test_df (DataFrame):  Dataset de test.
           timestamp (float):  Representación temporal en segundos.
           target (str):  Variable dependiente a usar.
           cols_to_remove (list): Columnas a retirar.

        Returns:
           DataFrame, DataFrame. Datasets de train y test para el modelo.
    """

    # Quitando columnas no usables
    print('------> Removing unnecessary columns')
    train_df = remove_unwanted_columns(train_df, cols_to_remove)
    test_df = remove_unwanted_columns(test_df, cols_to_remove)

    #Establezco como indice la columna 'index'
    train_df.set_index('index', inplace=True) 
    test_df.set_index('index', inplace=True)

    # guardando las columnas en IBM COS
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

