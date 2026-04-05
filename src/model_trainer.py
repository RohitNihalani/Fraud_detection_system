import os 
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_recall_curve, auc, classification_report

# Setup MLflow Tracking (Phase 5)
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Fraud_Detection_System")

class ModelTrainer:
    def train_model(self, X, y, preprocessor):
        # Phase 2: Stratified 80/20 split for imbalance handling [cite: 23]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Phase 4: Candidate architectures 
        models = {
            "Logistic_Regression": LogisticRegression(max_iter=1000),
            "Decision_Tree": DecisionTreeClassifier(criterion="gini"),
            "Random_Forest": RandomForestClassifier(n_estimators=100)
        }

        best_recall = 0
        best_pipeline = None

        for name, model_obj in models.items():
            # Phase 5: Start MLflow run for each model [cite: 43]
            with mlflow.start_run(run_name=name):
                pipeline = Pipeline([
                    ('preprocess', preprocessor),
                    ('model', model_obj)
                ])

                # Train
                pipeline.fit(X_train, y_train)
                
                # Predict
                y_pred = pipeline.predict(X_test)
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                
                # Phase 3 & 5: Strategic Metrics for Imbalance [cite: 31, 40]
                recall = recall_score(y_test, y_pred)
                precision, rec_curve, _ = precision_recall_curve(y_test, y_proba)
                pr_auc = auc(rec_curve, precision)
                
                # Log to MLflow [cite: 40]
                mlflow.log_params(model_obj.get_params())
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("pr_auc", pr_auc)
                
                # Log the model artifact to MLflow [cite: 42]
                mlflow.sklearn.log_model(pipeline, "model_pipeline")

                print(f"--- {name} ---")
                print(classification_report(y_test, y_pred))

                # Track best model for Phase 4 deliverable [cite: 34]
                if recall > best_recall:
                    best_recall = recall
                    best_pipeline = pipeline
                    # Save as best_model.pkl for Phase 6 API [cite: 34]
                    os.makedirs('models', exist_ok=True)
                    joblib.dump(pipeline, 'models/best_model.pkl')

        return best_pipeline