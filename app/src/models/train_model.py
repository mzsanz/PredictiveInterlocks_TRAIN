from ..data.make_dataset import make_dataset
from ..evaluation.evaluate_model import evaluate_model
from app import ROOT_DIR, cos, client
from sklearn.tree import DecisionTreeClassifier
from cloudant.query import Query
import time


def training_pipeline(path, model_info_db_name='predictive-interlocks-model'):
    """
        Function that implements the full training pipeline of the model.

        Args:
            path (str):  path to data.

        Kwargs:
            model_info_db_name (str):  database to use for storage of model info.
    """

    # Load the training configuration of the model
    model_config = load_model_config(model_info_db_name)['model_config']
    # dependent variable
    target = model_config['target']
    # columns to remove
    cols_to_remove = model_config['cols_to_remove']
    # columns to add
    cols_to_add = model_config['cols_to_add']

    # timestamp used for model and objects versioning
    ts = time.time()

    # load and transformation of the train and test dataset
    train_df, test_df = make_dataset(path, ts, target, cols_to_remove, cols_to_add)

    # split of variables: indepenedent and dependent 
    y_train = train_df[target]
    X_train = train_df.drop(columns=[target]).copy()
    y_test = test_df[target]
    X_test = test_df.drop(columns=[target]).copy()

    # model definition (Decision Tree Classifier)
    model = DecisionTreeClassifier(max_depth=model_config['max_depth'],
                                   min_samples_leaf=model_config['min_samples_leaf'],
                                   min_samples_split=model_config['min_samples_split'],
                                   random_state=50,
                                   n_jobs=-1)

    print('---> Training a model with the following configuration:')
    print(model_config)

    # Fit the model with the training dataset
    model.fit(X_train, y_train)

    # Saving the model in IBM COS
    print('------> Saving the model {} object on the cloud'.format('model_'+str(int(ts))))
    save_model(model, 'model',  ts)

    # Evaluating the model and collecting relevant info
    print('---> Evaluating the model')
    metrics_dict = evaluate_model(model, X_test, y_test, ts, model_config['model_name'])

    # Saving model info in documental database
    print('------> Saving the model information on the cloud')
    info_saved_check = save_model_info(model_info_db_name, metrics_dict)

    # Check of model info saving
    if info_saved_check:
        print('------> Model info saved SUCCESSFULLY!!')
    else:
        if info_saved_check:
            print('------> ERROR saving the model info!!')

    # Selection of the best model for production
    print('---> Putting best model in production')
    put_best_model_in_production(metrics_dict, model_info_db_name)


def save_model(obj, name, timestamp, bucket_name='uem-models-mzs'):
    """
        Function to store the model in IBM COS

        Args:
            obj (sklearn-object): trained model object
            name (str):  name of the object to use in the storing process
            timestamp (float):  time representation in seconds

        Kwargs:
            bucket_name (str):  IBM COS bucket to use.
    """
    cos.save_object_in_cos(obj, name, timestamp, bucket_name)


def save_model_info(db_name, metrics_dict):
    """
        Function to store model info in IBM Cloudant

        Args:
            db_name (str):  Database name.
            metrics_dict (dict):  Model info.

        Returns:
            boolean. Check if the document has been created.
    """
    db = client.get_database(db_name)
    client.create_document(db, metrics_dict)

    return metrics_dict['_id'] in db


def put_best_model_in_production(model_metrics, db_name):
    """
        Función para poner el mejor modelo en producción.

        Args:
            model_metrics (dict):  Info del modelo.
            db_name (str):  Nombre de la base de datos.
    """

    # conexión a la base de datos elegida
    db = client.get_database(db_name)
    # consulta para traer el documento con la info del modelo en producción
    query = Query(db, selector={'status': {'$eq': 'in_production'}})
    res = query()['docs']
    #  id del modelo en producción
    best_model_id = model_metrics['_id']

    # en caso de que SÍ haya un modelo en producción
    if len(res) != 0:
        # se realiza una comparación entre el modelo entrenado y el modelo en producción
        best_model_id, worse_model_id = get_best_model(model_metrics, res[0])
        # se marca el peor modelo (entre ambos) como "NO en producción"
        worse_model_doc = db[worse_model_id]
        worse_model_doc['status'] = 'none'
        # se actualiza el marcado en la BDD
        worse_model_doc.save()
    else:
        # primer modelo entrenado va a automáticamente a producción
        print('------> FIRST model going in production')

    # se marca el mejor modelo como "SÍ en producción"
    best_model_doc = db[best_model_id]
    best_model_doc['status'] = 'in_production'
    # se actualiza el marcado en la BDD
    best_model_doc.save()


def get_best_model(model_metrics1, model_metrics2):
    """
        Función para comparar modelos.

        Args:
            model_metrics1 (dict):  Info del primer modelo.
            model_metrics2 (str):  Info del segundo modelo.

        Returns:
            str, str. Ids del mejor y peor modelo en la comparación.
    """

    # comparación de modelos usando la métrica AUC score.
    auc1 = model_metrics1['model_metrics']['roc_auc_score']
    auc2 = model_metrics2['model_metrics']['roc_auc_score']
    print('------> Model comparison:')
    print('---------> TRAINED model {} with AUC score: {}'.format(model_metrics1['_id'], str(round(auc1, 3))))
    print('---------> CURRENT model in PROD {} with AUC score: {}'.format(model_metrics2['_id'], str(round(auc2, 3))))

    # el orden de la salida debe ser (mejor modelo, peor modelo)
    if auc1 >= auc2:
        print('------> TRAINED model going in production')
        return model_metrics1['_id'], model_metrics2['_id']
    else:
        print('------> NO CHANGE of model in production')
        return model_metrics2['_id'], model_metrics1['_id']


def load_model_config(db_name):
    """
        Función para cargar la info del modelo desde IBM Cloudant.

        Args:
            db_name (str):  Nombre de la base de datos.

        Returns:
            dict. Documento con la configuración del modelo.
    """
    db = client.get_database(db_name)
    query = Query(db, selector={'_id': {'$eq': 'model_config'}})
    return query()['docs'][0]
