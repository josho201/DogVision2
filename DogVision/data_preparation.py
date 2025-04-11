from pathlib import Path
import tarfile
import requests
from tqdm import tqdm   
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import tensorflow as tf
import scipy.io
import shutil
from DogVision.config import PATHS, MODEL_SETTINGS

class DataDownloader:
    def __init__(self):
        self.url =  PATHS.DATA_URL
        self.files = PATHS.FILES
        
    def download(self):
        for file in tqdm(self.files, desc="Downloading files"):
            self._download_file(file)
            
    def _download_file(self, filename):
        file_path = PATHS.RAW / filename
        if not file_path.exists():
            response = requests.get(f"{self.url}/{filename}", stream=True)
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            self._extract_tar(file_path)
            
    def _extract_tar(self, filepath):
        with tarfile.open(filepath) as tar:
            tar.extractall(path=PATHS.RAW)

class DataProcessor:
    def __init__(self):
        self.train_list = scipy.io.loadmat(PATHS.RAW / "train_list.mat")
        self.test_list = scipy.io.loadmat(PATHS.RAW / "test_list.mat")
        
    def create_splits(self):
        self._process_split(self.train_list, "train")
        self._process_split(self.test_list, "test")
        
    def _process_split(self, data_dict, split_name):
        dest_dir = PATHS.PROCESSED / split_name
        dest_dir.mkdir(exist_ok=True)
        
        for item in tqdm(data_dict["file_list"], desc=f"Processing {split_name}"):
            src_path = PATHS.RAW / "Images" / item[0][0]
            class_name = self._get_class_name(src_path.parent.name)
            dest_path = dest_dir / class_name / src_path.name
            
            dest_path.parent.mkdir(exist_ok=True)
            shutil.copy2(src_path, dest_path)
            
    def _get_class_name(self, folder_name):
        return "_".join(folder_name.split("-")[1:]).lower()
    
class Augmented: 
    
    def __init__(
            self,
            rotation = 20,
            width_shift = 0.1,
            height_shift = 0.1,
            horizontal_flip = True
            ):
        
        self.train_gen = ImageDataGenerator(
            rotation_range=rotation,
            width_shift_range=width_shift,
            height_shift_range=height_shift,
            horizontal_flip=horizontal_flip,
            
        ) 

    def create_augmented_train(self, rounds =1):
    
        self.train_generator = self.train_gen.flow_from_directory(
            PATHS.TRAIN_DIR,  # Folder with subfolders per class
            target_size=MODEL_SETTINGS.IMG_SIZE,
            batch_size= MODEL_SETTINGS.BATCH_SIZE,
            class_mode='categorical', # or 'binary'
            save_to_dir = PATHS.AUGMENTED_TRAIN
        )

        self.train_10_percent = self.train_gen.flow_from_directory(
            PATHS.TRAIN_10_DIR,  # Folder with subfolders per class
            target_size=MODEL_SETTINGS.IMG_SIZE,
            batch_size= MODEL_SETTINGS.BATCH_SIZE,
            class_mode='categorical'  ,# or 'binary',
            save_to_dir = PATHS.AUGMENTED_TRAIN_10
        )

        self.val_datagen = ImageDataGenerator()  # No augmentation for validation

        self.val_data = self.val_datagen.flow_from_directory(
            PATHS.TEST_DIR,
            target_size=MODEL_SETTINGS.IMG_SIZE,
            batch_size= MODEL_SETTINGS.BATCH_SIZE,
            class_mode='categorical'
        )

        steps_10 = self.train_10_percent.samples // self.train_10_percent.batch_size
        steps_train = self.train_generator.samples // self.train_generator.batch_size

        for i in range(rounds):
            for i in range(steps_train):
                 _ = next(self.train_10_percent)
            for i in range(steps_10):
                _ = next(self.train_10_percent)

    def create_training_dataset(self):


        # Create train 10% dataset
        self.train_10_percent_ds = tf.keras.utils.image_dataset_from_directory(
            directory= PATHS.TRAIN_10_DIR,
            label_mode="categorical", # turns labels into one-hot representations (e.g. [0, 0, 1, ..., 0, 0])
            batch_size=MODEL_SETTINGS.BATCH_SIZE,
            image_size=MODEL_SETTINGS.IMG_SIZE,
            shuffle=True, # shuffle training datasets to prevent learning of order
            seed=MODEL_SETTINGS.SEED
        )

        # Create full train dataset
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=PATHS.TRAIN_DIR,
            label_mode="categorical",
            batch_size=MODEL_SETTINGS.BATCH_SIZE,
            image_size=MODEL_SETTINGS.IMG_SIZE,
            shuffle=True,
            seed=MODEL_SETTINGS.SEED
        )

        # Create test dataset
        self.test_ds = tf.keras.utils.image_dataset_from_directory(
            directory=PATHS.TEST_DIR,
            label_mode="categorical",
            batch_size=MODEL_SETTINGS.BATCH_SIZE,
            image_size=MODEL_SETTINGS.IMG_SIZE,
            shuffle=False, # don't need to shuffle the test dataset (this makes evaluations easier)
            seed=MODEL_SETTINGS.SEED

        )

Dataset = Augmented()