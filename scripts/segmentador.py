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
    x = tf.image.resize(array_img[None, ...],(resize,resize),method='bilinear',antialias=True)
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

if __name__ == "__main__":
    main()