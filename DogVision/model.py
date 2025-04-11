import tensorflow as tf
from keras import layers, Model

def create_model(
        num_classes, 
        input_shape: tuple[int, int, int]=(224, 224, 3), 
        dropout=0.5, 
        trainable = False,
        augmentation = False,
    ):




# Create data augmentation layer
    

    
    
    inputs = layers.Input(shape=input_shape)

    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
      
    )
    base_model.trainable = trainable
    
    x = base_model(inputs)
    if(augmentation):
        data_augmentation_layer = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"), # randomly flip image across horizontal axis
            layers.RandomRotation(factor=0.4), # randomly rotate image
            # layers.RandomBrightness(factor = 0.2),
            #layers.RandomZoom(height_factor=0.3, width_factor=0.3) # randomly zoom into image
            # More augmentation can go here
        ],
        name="data_augmentation"
        )
        x = data_augmentation_layer(inputs)

    

    x = layers.GlobalAveragePooling2D()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs, outputs)