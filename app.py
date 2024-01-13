import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow, pickle
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path
import os
from dotenv import load_dotenv
import joblib
from sklearn.naive_bayes import BernoulliNB

from log_mlflow_model.main import mlflow_demo


# Rest of your code



# Your main program goes here
def main():
    try:
        object = mlflow_demo()
        object.log_mlfow_model()
    except Exception as e:
        raise e    



if __name__ == "__main__":
    # Execute the main program

    load_dotenv('mlflow.env')
     # Access environment variables "key" using os.getenv to fetch "value"
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
     # setting up the mlflow regirsty with our tracking uri 
    mlflow.set_registry_uri(MLFLOW_URI)
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    print(f"tracking_url_type_store in app.py: {tracking_url_type_store}")
    main()
