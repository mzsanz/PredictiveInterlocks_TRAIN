from flask import Flask
import os
from app.src.models import train_model
from app import ROOT_DIR
import warnings

# Remove unneeded warnings
warnings.filterwarnings('ignore')

# start-up de app under the Flask framework
app = Flask(__name__)

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8000))


# using @app.route to manage routers (GET method)
# root path "/"
@app.route('/', methods=['GET'])
def root():
    """
        Function to manage the output from root path.

        Returns:
           dict.  Output message
    """
    # Do nothing. Just return info
    return {'Project':'Predictive Interlocks'}


# path to start de training pipeline (GET method)
@app.route('/train-model', methods=['GET'])
def train_model_route():
    """
        Function to start-up the training pipeline

        Returns:
           dict.  Output message
    """
    # Path to load local data
    df_path = os.path.join(ROOT_DIR, 'data/project_dataset.csv')

    # Start the training pipeline of our model
    train_model.training_pipeline(df_path)

    # Return message
    return {'TRAINING MODEL': 'Predictive Interlocks'}


# main
if __name__ == '__main__':
    # execution of the app
    app.run(host='0.0.0.0', port=port, debug=True)
