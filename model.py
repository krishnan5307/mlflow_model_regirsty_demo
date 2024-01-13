import logging
import joblib
import mlflow, os
from urllib.parse import urlparse
MLFLOW_URI = r"https://dagshub.com/krishnan5307/Credit_card_default_prediction_with_mlflow.mlflow"
     # setting up the mlflow regirsty with our tracking uri 
mlflow.set_registry_uri(MLFLOW_URI)
tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
from dotenv import load_dotenv


class mlflow_demo:
            
      def __init__(self):
                try:
                        pass
                          
                      #  logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
                                  
                except Exception as e:
                    raise e
                

      def log_mlfow_model(self):
            
            try:

                print("Started Main program execution")
                file_path_obj = r"C:\data science\MLops\mlflow_model_regirsty_demo\model_objects\model_obj.pkl"
                file_path_model = r"C:\data science\MLops\mlflow_model_regirsty_demo\model\model.pkl"
                file_path_model_factory = r"C:\data science\MLops\mlflow_model_regirsty_demo\mode_object_factory\LogisticRegression(C=10.0).joblib"
                    
                # df= pd.read_csv(r"credit_card.csv", )                                           ## using dill library to load file
                with open(file_path_model_factory, "rb") as file_obj:
                        model= joblib.load(file_obj)
            #         pred = model.predict(df)

            #         pred_df = pd.DataFrame({'Column_Name': pred})
            #         # Save the DataFrame to a CSV file
            #         pred_df.to_csv("pred_df", index=False)
            # # Log the CSV file as an artifact
            #         mlflow.log_artifact("pred_df")
                        

                print(model)
            #     model = BernoulliNB(alpha=1.0, binarize=0.0)
                print(f"tracking_url_type_store inside function call: {tracking_url_type_store}")
                with mlflow.start_run():
                        
                        mlflow.sklearn.log_model(model, "model") 
                            
                 #   predicted_qualities = model.predict(test_x)
                     #   Model registry does not work with file store
                        if tracking_url_type_store != "file":

                            # Register the model
                            # There are other ways to use the Model Registry, which depends on the use case,
                            # please refer to the doc for more information:
                            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                            mlflow.sklearn.log_model(model, "model", registered_model_name="BESTModelinstance")
                        else:
                            mlflow.sklearn.log_model(model, "model")    

            except Exception as e:
                raise e
            



if __name__ == "__main__":
    # Execute the main program

    load_dotenv(r"C:\\data science\\MLops\\mlflow_model_regirsty_demo\\mlflow.env")
     # Access environment variables "key" using os.getenv to fetch "value"
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
     # setting up the mlflow regirsty with our tracking uri 
    mlflow.set_registry_uri(MLFLOW_URI)
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    print(f"tracking_url_type_store in __main__: {tracking_url_type_store}")

    file_path_model_factory = r"C:\data science\MLops\mlflow_model_regirsty_demo\mode_object_factory\LogisticRegression(C=10.0).joblib"
                    
                # df= pd.read_csv(r"credit_card.csv", )                                           ## using dill library to load file
    with open(file_path_model_factory, "rb") as file_obj:
                        model= joblib.load(file_obj)
   
    with mlflow.start_run():
          mlflow.sklearn.log_model(model,"model", registered_model_name="BESTModelinstance")
    # ob = mlflow_demo()
    # ob.log_mlfow_model()
