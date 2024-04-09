import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.utils import load_obj

import pandas as pd

class PredictPipeline:
    def __init__(self) -> None:
        pass
    
    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            #preprosessor is transformed using features
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("error ocured in prediction function in prediction_pipleline")
            raise CustomException(e,sys)

class CustomData:
    def __init__(self, carat:float, depth:float, table:float, x:float, y:float, z:float, cut:str, color:str, clarity:str):
        self.carat =carat
        self.depth =depth
        self.table = table
        self.x=x
        self.y = y
        self.z =z
        self.cut= cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_inpt_dic = {
                'carat': [self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_inpt_dic)
        except Exception as e:
            logging.info("error ocured in get_data_as_dataframe function in prediction_pipleline")
            raise CustomException(e,sys)

   