from src.data_loader import Data_Loader
from src.data_preprocessing import Datapreprocessing
from src.model_trainer import ModelTrainer

class TrainPipeline:
    def run_pipeline(self):
        print("--- Pipeline Started ---") 
        ingestion = Data_Loader()
        data = ingestion.load_data(r'C:\Users\rohit\OneDrive\Desktop\fraud-detection\data\Synthetic_Financial_datasets_log.csv')
        print("Data loaded successfully.")

        preprocessing = Datapreprocessing()
        X, y, preprocessor = preprocessing.preprocessdata(data)
        print("Preprocessing complete.")

        trainer = ModelTrainer()
        print("Starting model training...")
        trainer.train_model(X, y, preprocessor)
        print("--- Pipeline Finished Successfully ---")

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline() 