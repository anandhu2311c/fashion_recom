import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from skimage.io import imread
from skimage.transform import resize
import os
import pandas as pd
import tensorflow as tf
import keras

# ------------------- Define Encoder-Decoder (Autoencoder) -------------------
input_img = Input(shape=(320, 192, 3))

x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')

# ------------------- Load Cropped Images from real_fashion_data.csv -------------------
df = pd.read_csv("/content/real_fashion_data.csv")
image_folder = "/content/crop/cropped_tshirts"

images = []
for i in range(len(df)):
    img_path = os.path.join(image_folder, f"tshirt_{i}_0.jpg")
    if os.path.exists(img_path):
        try:
            img = imread(img_path)
            img = resize(img, (320, 192, 3))
            images.append(img)
        except:
            continue

print(f"Loaded {len(images)} cropped t-shirt images for training.")

x_data = np.array(images).astype('float32')

# Split into train and test
split_idx = int(0.85 * len(x_data))
x_train = x_data[:split_idx]
x_test = x_data[split_idx:]

# ------------------- Train Encoder -------------------
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=32,
                validation_data=(x_test, x_test))


# ------------------- Save Encoder Only -------------------
encoder = Model(input_img, encoded)
os.makedirs("pm_model_components", exist_ok=True)
encoder.save("pm_model_components/encoder.keras")  # or "encoder.h5"

