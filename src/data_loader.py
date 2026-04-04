import pandas as pd

class Data_Loader:
    def load_data(self,path):
        data=pd.read_csv(path)
        return data