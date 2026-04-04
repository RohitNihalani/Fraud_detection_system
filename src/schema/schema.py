from pydantic import BaseModel

class BankData(BaseModel):
    step:int
    type:str
    amount:float
    oldbalanceOrg:float
    newbalanceOrig:float
    oldbalanceDest:float
    newbalanceDest:float