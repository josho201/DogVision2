from pathlib import Path
# Define the targetimages_split_dir = 
# Define the training and test directories


class PathConfig:
    def __init__(self):
        self.IMAGES_SPLIT_DIR = Path("images_split")
        self.ROOT = Path(__file__).resolve().parents[2]

        self.TRAIN_DIR = self.IMAGES_SPLIT_DIR / "train"
        self.TEST_DIR = self.IMAGES_SPLIT_DIR / "test"
        self.TRAIN_10_DIR = self.IMAGES_SPLIT_DIR / "train_10_percent"


        self.AUGMENTED = Path("augmented_images")
        self.AUGMENTED_TRAIN = self.AUGMENTED / "train"
        self.AUGMENTED_TRAIN_10 = self.AUGMENTED / "train_10"


        self.MODELS = self.ROOT / "models"

        self.DATA_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs" 
        self.FILES =  ["images.tar", "annotation.tar", "lists.tar"]
       

        self.AUGMENTED.mkdir(exist_ok=True)
        self.AUGMENTED_TRAIN_10.mkdir(exist_ok=True)
        self.AUGMENTED_TRAIN.mkdir(exist_ok=True)

        self.MODELS.mkdir(exist_ok=True)

class ModelConfig:
    def __init__(self):
        self.BATCH_SIZE = 32
        self.SEED = 4
        self.IMG_SIZE = (224,224)
        self.LEARNING_RATE = 0.001

MODEL_SETTINGS = ModelConfig()
PATHS = PathConfig()


print(PATHS.MODELS)