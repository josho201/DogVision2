import numpy as np
from pathlib import Path
import tensorflow as tf

def count_parameters(model):
    trainable = np.sum([np.prod(layer.shape) for layer in model.trainable_weights])
    non_trainable = np.sum([np.prod(layer.shape) for layer in model.non_trainable_weights])
    return trainable, non_trainable

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs available: {len(gpus)}")
        except RuntimeError as e:
            print(e)