from DogVision.data_preparation import Dataset
from DogVision.trainer import DogVisionTrainer
from DogVision.eval import ModelEvaluator
from DogVision.utils import setup_gpu
# from DogVision.gradio_app import launch_gradio

def main():
    setup_gpu()
    
    # Download and process data
    """
    DataDownloader().download()
    DataProcessor().create_splits()
    """
    
    # Train mode
    Dataset.create_training_dataset()
    train =Dataset.train_ds
    val = Dataset.test_ds

    trainer = DogVisionTrainer(num_classes=120, train_data=train, val_data= val)
    history = trainer.train_dv(epochs=12)
 
    
    """
    # Evaluate model
    evaluator = ModelEvaluator(trainer.model, trainer.val, trainer.class_names)
    evaluator.evaluate()
    evaluator.plot_confusion_matrix()
     """
    
     
    # launch_gradio()
if __name__ == "__main__":
    main()