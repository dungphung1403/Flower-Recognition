
import os
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator , load_img
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt


# Delete corrupted images  
num_skipped = 0
dirs = os.listdir('train/')
for dir in dirs:
    files = os.listdir('train/' + dir)
    for file in files:
        fpath = os.path.join('train/',dir, file)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print(f"Deleted {num_skipped} images.")


# Preparing Dataset
print("Preparing Dataset...")
base_dir = 'train/'
img_size = 180
batch_size = 32


train_ds = tf.keras.utils.image_dataset_from_directory(base_dir, 
                                                        validation_split=0.2,
                                                        subset="training",
                                                        seed=123,                                                        
                                                        batch_size=batch_size,
                                                        image_size =(img_size, img_size)
                                                        )
 
val_ds = tf.keras.utils.image_dataset_from_directory(base_dir, 
                                                        validation_split=0.2,
                                                        subset="validation",
                                                        seed=123,                                                       
                                                        batch_size=batch_size,
                                                        image_size = (img_size, img_size)
                                                        )   


AUTOTUNE = tf.data.AUTOTUNE
train_ds =  train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds =  val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#Data augmentation
print("Building Model...")
data_augmentation = Sequential([
        layers.RandomFlip("horizontal", input_shape=(img_size, img_size,3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
])

# training the model
model =  Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5)
    ]
)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy']
                )


history = model.fit(train_ds, epochs=15, validation_data=val_ds)

#saving the model
print("Saving Model...")
with open('flower_names.bin', 'wb') as f_out:
    pickle.dump(model, f_out)





