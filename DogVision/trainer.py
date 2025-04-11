from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from DogVision.config import PATHS, MODEL_SETTINGS
from DogVision.model import create_model
import tensorflow as tf
class DogVisionTrainer:
    def __init__(
            self, 
            num_classes,
            train_data, 
            val_data,
            batch_size=32, 
            img_size=(224, 224)   
            ):
        
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.img_size = img_size
        self.train = train_data
        self.val = val_data

        self.model = create_model(self.num_classes)
        
    def train_dv(self, epochs=10):
        self.model.compile(
            optimizer= tf.keras.optimizers.Adam(learning_rate = MODEL_SETTINGS.LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        callbacks = [
            EarlyStopping(patience=3),
            ModelCheckpoint(
                str(PATHS.MODELS / "best_model.keras"),
                save_best_only=True
            )
        ]
        
        history = self.model.fit(
            self.train,
            epochs=epochs,
            validation_data=self.val,
            callbacks=callbacks
        )
        
        return history