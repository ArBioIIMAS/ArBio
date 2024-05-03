# import os
# os.environ["TF_USE_LEGACY_KERAS"] = "1"
# import tf_keras as keras

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import preprocessing
from model import modelo
import io


resize = 512
st.title("Deep learning-based histopathological segmentation")
st.header("Load an image")
st.set_page_config(page_title="ArBio: Nest segmentation",
                   page_icon="🧊",
                   layout="wide",
                   initial_sidebar_state="expanded",)

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        fig = plt.figure()
        plt.imshow(image)
        plt.axis("off")
        st.pyplot(fig)
        predictions = predict(image)
        st.write(predictions)
        
def predict(image):
    IMAGE_SHAPE = (resize, resize,3)
    model = modelo() #load model
    img = image.convert('RGB')
    array_img = np.asarray(img)/255
    #x = tf.image.resize(array_img[None, ...],(resize,resize),method='bilinear',antialias=True)

    x = keras.image.resize(array_img[None, ...],(resize,resize),method='bilinear',antialias=True)

    mask_array = np.asarray(model.predict(x)[0, ..., 0]*255)

    original_shape = array_img.shape
    output_shape = mask_array.shape

    #Calling to segmentation process
    st.header("Masking")
    encode_mask(mask_array)
    result = f"Input shape: {original_shape},  ouput shape: {output_shape}"
    return result

def encode_mask(mask_array):
    with io.BytesIO() as bimg:
        new_mask = Image.fromarray(mask_array.astype(np.uint8), 'L')
        fig = plt.figure()
        plt.imshow(new_mask)
        plt.axis("off")
        st.pyplot(fig)    

def modelo():
    # U-NET Model
    inputs = keras.layers.Input((512,512,3))
    
    #Contraction path
    c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = keras.layers.Dropout(0.1)(c1)
    c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    
    c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = keras.layers.Dropout(0.1)(c2)
    c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = keras.layers.Dropout(0.2)(c3)
    c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = keras.layers.Dropout(0.2)(c4)
    c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = keras.layers.Dropout(0.3)(c5)
    c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = keras.layers.Dropout(0.2)(c6)
    c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = keras.layers.Dropout(0.2)(c7)
    c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = keras.layers.Dropout(0.1)(c8)
    c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = keras.layers.concatenate([u9, c1], axis=3)
    c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = keras.layers.Dropout(0.1)(c9)
    c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    unet = keras.Model(inputs=[inputs], outputs=[outputs])

    unet.load_weights('pesos_chagas') #load_weights

    return unet



if __name__ == "__main__":
    main()