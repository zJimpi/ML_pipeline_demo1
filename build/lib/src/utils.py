import os 
import sys   
import pickle     
from src.exception import CustomException
from sklearn.metrics import r2_score


# Save function
def save_function(file_path, obj): 
    dir_path = os.path.join(file_path) #path =artifacts/preprocesor.pkl 
    os.makedirs(dir_path, exist_ok= True) #making the directory
    with open(file_path, "wb") as file_obj: #opening as write byte mode
        pickle.dump(obj, file_obj) #saving preprocesor data in pkl file

def load_obj(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)

def model_performance(X_train,y_train, X_test, y_test, models): 
    try: 
        report = {}
        for i in range(len(models)): 
            model = list(models.values())[i]  #i-th element from the list of model objects.
            model.fit(X_train, y_train)
            # Train models
            y_test_pred = model.predict(X_test)
            # Test data
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score # storing model name as the key and the R2 score as the value
        return report
    except Exception as e:  
        raise CustomException(e,sys) 

