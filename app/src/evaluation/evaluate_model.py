from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from datetime import datetime


def evaluate_model(model, X_test, y_test, timestamp, model_name):
    """
        This function evaluates the trained model and creates a dictionary with all its info

        Args:
           model (sklearn-object):  Trained model object.
           X_test (DataFrame): Independent test variables.
           y_test (Series):  Dependent test variables.
           timestamp (float):  Time in seconds.
           model_name (str):  Model name

        Returns:
           dict. Dictionary with the model info
    """

    # compute predictions with the trained model
    y_pred = model.predict(X_test)

    # create a dictionary with the model info
    model_info = {}
    
    # general model info
    model_info['_id'] = 'model_' + str(int(timestamp))
    model_info['name'] = 'model_' + str(int(timestamp))
    # training timestamp (dd/mm/YY-H:M:S)
    model_info['date'] = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    model_info['model_used'] = model_name
    # objects used in the model (imputer and scaler)
    model_info['objects'] = {}
    model_info['objects']['imputer'] = 'imputer_' + str(int(timestamp))
    model_info['objects']['scaler'] = 'scaler_' + str(int(timestamp))
    # metrics
    model_info['model_metrics'] = {}
    model_info['model_metrics']['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    model_info['model_metrics']['accuracy_score'] = accuracy_score(y_test, y_pred)
    model_info['model_metrics']['precision_score'] = precision_score(y_test, y_pred)
    model_info['model_metrics']['recall_score'] = recall_score(y_test, y_pred)
    model_info['model_metrics']['f1_score'] = f1_score(y_test, y_pred)
    model_info['model_metrics']['roc_auc_score'] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    # model status (in production or not)
    model_info['status'] = "none"

    return model_info



