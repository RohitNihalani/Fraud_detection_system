import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

class Datapreprocessing:
    def preprocessdata(self,data):
         X=data.drop(['nameOrig','nameDest','isFraud','isFlaggedFraud'],axis=1)
         y=data['isFraud']

         categorical=X.select_dtypes(include='object').columns
         numerical=X.select_dtypes(exclude='object').columns

         preprocessor=ColumnTransformer(
            transformers=[
            ('nums',StandardScaler(),numerical),
            ('cats',OneHotEncoder(drop='first'),categorical)
        ])

         return X,y,preprocessor