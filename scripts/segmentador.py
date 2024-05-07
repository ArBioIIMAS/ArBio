import streamlit as st
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import preprocessing
#from model import modelo
import io

resize = 512
st.title("Deep learning-based histopathological segmentation")
st.header("Load an image")

def load_model():
    import urllib.request
    import os
    if not os.path.isfile('model.h5'):
        urllib.request.urlretrieve('https://github.com/ArBioIIMAS/ArBio/raw/main/scripts/pesos_chagas.h5', 'model.h5')
    return tf.keras.models.load_model('model.h5')

def main():
    
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    if file_uploaded is not None:
        print("Loading image")
        image = Image.open(file_uploaded)
        fig = plt.figure()
        plt.imshow(image)
        plt.axis("off")
        st.pyplot(fig)

        print("Segmentation")
        model = load_model()
        predictions = predict(model,image)
        #st.write(predictions)

   
def predict(model, image):
    IMAGE_SHAPE = (resize, resize,3)
    #model = modelo() #load model
   
    img = image.convert('RGB')
    array_img = np.asarray(img)/255
    x = tf.image.resize(array_img[None, ...],(resize,resize),method='bilinear',antialias=True)
    #mask_array = np.asarray(model.predict(x)[0, ..., 0]*255)

    predictions = model.predict_generator(image, verbose=1)
    mask_array = np.asarray(predictions*255)
    return mask_array


def encode_mask(mask_array):
    with io.BytesIO() as bimg:
        new_mask = Image.fromarray(mask_array.astype(np.uint8), 'L')
        fig = plt.figure()
        plt.imshow(new_mask)
        plt.axis("off")
        st.pyplot(fig)

        st.header("Binary segmentation mask")
        import cv2
        _, thresh2 = cv2.threshold(mask_array, 120, 255, cv2.THRESH_BINARY) 

        fig = plt.figure()
        plt.imshow(thresh2,cmap="gray")
        plt.axis("off")
        st.pyplot(fig)  

if __name__ == "__main__":
    main()