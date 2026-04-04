import os 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,classification_report,accuracy_score

class ModelTrainer:
    def train_model(self,X,y,preprocessor):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        pipeline=Pipeline([
            ('preprocess',preprocessor),
            ('model',DecisionTreeClassifier(criterion="gini"))
        ])

        pipeline.fit(X_train,y_train)
        y_pred=pipeline.predict(X_test)
        
        print(classification_report(y_test,y_pred))

        print(f1_score(y_test,y_pred))

        os.makedirs('models',exist_ok=True)
        joblib.dump(pipeline,'models/model.pkl')

        return pipeline