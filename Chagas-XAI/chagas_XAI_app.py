"""
Para correr: 
streamlit run chagas_XAI_app.py
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, resample_poly

import streamlit as st
from pypots.optim import Adam
import torch.nn.functional as F
from pypots.classification import TimesNet
from captum.attr import IntegratedGradients

from PIL import Image
import io
import urllib.request
import os
#****************************************************************#
resize = 512
width = 100
height = 100
device = ('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
path_model = "bs_64_TimesNet_fs_no_aug_no.pypots"
lead_names_ptb = ["DI", "DII", "DIII", "AVL", "AVR", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
lead_names = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"] # mismo que samitrop!

feature_selection = "si" #minúsculas
lead_names_ptb = ["DI", "DII", "DIII", "AVL", "AVR", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
lead_names = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"] # mismo que samitrop!
lead_names_FE = ['AVR', 'DI', 'DII', 'V1', 'AVF']

#************************** Functions ***************************#
def load_logo():
    if not os.path.isfile('logo_arbio.png'):
        urllib.request.urlretrieve('https://arbioiimas.github.io/ArBio/images/logo_arbio.png', 'logo_arbio.png')
    return Image.open('logo_arbio.png')

def add_logo(width, height):
    """Read and return a resized logo"""
    logo = load_logo()
    modified_logo = logo#.resize((width, height))
    return modified_logo

def predict(model, image):
    IMAGE_SHAPE = (resize, resize,3)
   
    img = image.convert('RGB')
    array_img = np.asarray(img)/255
    x = tf.image.resize(array_img[None, ...],(resize,resize),method='bilinear',antialias=True)

    predictions = model.predict_generator(x, verbose=1)
    mask_array = np.asarray(predictions[0, ..., 0]*255)

    st.divider()
    st.header("Nest probability map")
    encode_mask(mask_array)
    st.divider()
    st.header("Binary segmentation mask")
    binary_mask(mask_array)

    result = "To save the mask, just right-click on image."
    return result

def encode_mask(mask_array):
    with io.BytesIO() as bimg:
        new_mask = Image.fromarray(mask_array.astype(np.uint8), 'L')
        fig = plt.figure()
        plt.imshow(new_mask)
        plt.axis("off")
        st.pyplot(fig, bbox_inches='tight', pad_inches=0) 

def binary_mask(mask_array):
    with io.BytesIO() as bimg:
        import cv2
        r, thresh2 = cv2.threshold(mask_array, 120, 255, cv2.THRESH_BINARY)
        fig = plt.figure()
        plt.imshow(thresh2,cmap="gray")
        plt.axis("off")
        st.pyplot(fig, bbox_inches='tight', pad_inches=0) 


def bandpass_filter(signal, fs, low=0.5, high=40):
    """Filtro pasa-banda """
    nyq = 0.5 * fs
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def resample_signal(signal, fs_orig, fs_target):
    """ Remuestreo """
    if fs_orig == fs_target:
        return signal
    up = 4
    down = 5
    return resample_poly(signal, up, down)

def normalize(signal):
    """ Normalización (Z-score) """
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

def preprocess(signals, n_steps, n_features): # signals, fs_orig, fs_target, database
    print("Before processing (original): ", signals.shape) #(num_patients, leads, amplitud) (5,12,4096)

    if n_features == 12: 
        DF_all_patients = np.zeros((1,len(lead_names),4000))
    else: 
        DF_all_patients = np.zeros((1,len(lead_names_FE),4000))

    if n_steps > 4096: 
        fs_orig = 500
        fs_target = 400
    else: 
        fs_orig = 400
        fs_target = 400

    print("DF_all_patients: ", DF_all_patients.shape, "fs_orig: ", fs_orig)

    for id_pat in range(signals.shape[0]):
        selected_patient = signals[id_pat, :, :] #(12, 5000)
        print("selected_patient: ", selected_patient.shape)

        signals_transpose = selected_patient.T #(5000,12)
        print("selected_patient: ", selected_patient.shape)
        df_ecg = pd.DataFrame(signals_transpose, columns = lead_names)#(5000,12)
        print("df_ecg: ", df_ecg.shape)

        DF_per_patient = pd.DataFrame()
        for lead in lead_names:
            signal = df_ecg[lead].to_numpy() #(5000)
            print("signal: ", signal.shape)
            signal = bandpass_filter(signal, fs_orig)#(5000)
            print("bandpass_filter: ", signal.shape)
            signal = resample_signal(signal, fs_orig, fs_target) #(4000)
            print("resample_signal: ", signal.shape)
            signal = normalize(signal) #(4000)  
            print("normalize: ", signal.shape)
            DF_signal = pd.DataFrame(signal,columns=[lead]) #(4000, 1)
            print("DF_signal: ", DF_signal.shape)   
            DF_per_patient = pd.concat([DF_signal, DF_per_patient], axis = 1)
            print("DF_per_patient: ", DF_per_patient.shape) 
            print("------------")
        
        if n_features == 5:  
            DF_per_patient = DF_per_patient[lead_names_FE] 
        DF_per_patient_numpy = DF_per_patient.to_numpy().T #(4000,12)
        DF_per_patient_numpy = DF_per_patient_numpy[np.newaxis, :, :] #[1,12,4000]
        DF_all_patients = np.concatenate((DF_per_patient_numpy, DF_all_patients), axis=0) #(3, 12, 5000)
    x = DF_all_patients[:-1] #(num_patients, num_leads, freq_muestreo)
    print("x: ", x.shape)
    return x, x.shape[2], x.shape[1]

def loading_signals(path_matlab):#folder, time_steps, database
    data_ecg = loadmat(path_matlab)
    signals = data_ecg['muestra'] #(12, 4096)
    signals_transpose = signals.T #(4096,12)
    print("signals_transpose: ", signals_transpose.shape)

    if  signals_transpose.shape[0] > 4096: 
        df_ecg = pd.DataFrame(signals_transpose, columns = lead_names_ptb) #(4096, 12)
        df_ecg = df_ecg[lead_names] 
        print("\n\nPTB-XL:", df_ecg.shape) #(5000, 12)
        y = 0
    
    else: #es SAMITROP (4096)
        signals_transpose = signals_transpose[48:4048] #retomado únicamente 4000 pasos de tiempo
        df_ecg = pd.DataFrame(signals_transpose, columns = lead_names) #(4000, 12)
        print("\n\nSAMITROP:", df_ecg.shape) #(4000, 12)
        y = 1
    
    df_ecg = df_ecg.to_numpy().T #[12,4000]
    df_ecg = df_ecg[np.newaxis, :, :] #[1,12,4000]
    print("df_ecg: ", df_ecg.shape, "y:", y)
    return df_ecg, df_ecg.shape[2], df_ecg.shape[1], y

def get_model(n_steps, n_features):
    model = TimesNet(
                n_steps=n_features,
                n_features= n_steps, 
                n_classes=2,
                n_layers = 5,
                top_k = 2,
                d_model = 32, #256
                d_ffn = 64,
                n_kernels = 5,
                dropout = 0.8,
                batch_size=batch_size,
                epochs=200,
                patience=20,
                optimizer=Adam(lr=1e-3),
                num_workers=0,
                device=device)
    model.load(path_model)
    return model

def get_importance(signal, logits, pred_class, model, n_steps, n_features, ytrue):
    signal_tensor = torch.from_numpy(signal).to(torch.float32).to(device) #torch.Size([1, 5, 4096])
    signal_tensor.requires_grad = True
    baseline = torch.zeros_like(signal_tensor, dtype=torch.float).to(device) #[1,5,4096]


    def forward_func(x):
        model.eval()
        output = model({"X": x})
        logits = output["logits"]
        return logits  

    ig = IntegratedGradients(forward_func, multiply_by_inputs=False) ## defining and applying integrated gradients
    attributions = ig.attribute(signal_tensor,baselines=baseline,target=pred_class,n_steps=100) #torch.Size([1, 5, 4000])

    signal = signal_tensor.squeeze(0).detach().cpu().numpy() #(5, 4000)
    attr = attributions.squeeze(0).detach().cpu().numpy() #(5, 4000)

    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities.cpu().detach().numpy().flatten().tolist() #[0.6005609631538391, 0.3994390368461609]

    if n_features == 12:
        features_names = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    else:
        features_names = ["DI", "DII", "AVR", "AVF", "V1"]
    
    print("********************************************")
    print("ytrue: ", ytrue, "ypred: ", pred_class.item())

    if pred_class.item()==1:
        print("** Positive", "Ytrue: ", ytrue, "Ypredicted: ", pred_class.item(), "probabilities: ", probabilities)
        plot_12lead_heatmap(signal,attr,features_names, probabilities[1], "positive", sampling_rate=400,cmap='turbo')

    else:
        print("** Negative","Ytrue: ", ytrue, "Ypredicted: ", pred_class.item(), "probabilities: ", probabilities)
        plot_12lead_heatmap(signal,attr,features_names, probabilities[0], "negative", sampling_rate=400)

def plot_12lead_heatmap(signal,attr,lead_names, probabilidades, label, sampling_rate=400,cmap='turbo'):
    fig, axes = plt.subplots(len(lead_names),1,figsize=(16, 12),sharex=True)
    attr = np.maximum(attr, 0)
    for i, ax in enumerate(axes):
        sig = signal[i]
        at = np.abs(attr[i])
        at = gaussian_filter1d(at, sigma=50)
        at = at / (at.max() + 1e-8)
        t = np.arange(len(sig)) / sampling_rate
        heatmap = np.tile(at, (120, 1))
        y_min = sig.min() - 0.3
        y_max = sig.max() + 0.3

        if label =="positive":
            ax.imshow(heatmap,extent=[t.min(), t.max(), y_min, y_max],aspect='auto',origin='lower',cmap=cmap,alpha=0.6)
        else:
            ax.imshow(heatmap,extent=[t.min(), t.max(), y_min, y_max],aspect='auto',origin='lower',alpha=0.6)
        line, = ax.plot(t,sig,color='deepskyblue',linewidth=1.7)
        ax.set_ylabel(lead_names[i],rotation=0,fontsize=8,labelpad=18)


        line.set_path_effects([pe.Stroke(linewidth=4, foreground='white'),pe.Normal()])

        ax.set_yticks([])
        ax.spines[['top', 'right', 'left']].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    st.pyplot(fig, bbox_inches='tight', pad_inches=0) 

def main():
    file_uploaded = st.file_uploader("Choose File", type=["hdf5", "mat"])

    if file_uploaded is not None:
        print("Loading signals")
        signals, n_steps, n_features, ytrue = loading_signals(file_uploaded) #signals: (1, 12, 5000) n_steps: 5000 n_features:  12
        signals, n_steps, n_features = preprocess(signals, n_steps, n_features) #(1, 12, 4000)

        print("Prediction")
        print("\n\n ************************** Loading model **************************")
        model = get_model(n_steps,n_features)

        predictions = model.predict({"X": signals})
        model_prediction = predictions["classification"] #[1]
        logits = torch.from_numpy(predictions["logits"]) #tensor([[-3.5400,  3.3134]])
        pred_class = torch.argmax(logits, dim=1) #tensor([1])

        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities.cpu().detach().numpy().flatten().tolist()

        print("**************************************")
        print("\n\n model_prediction: ", model_prediction)
        print("\n\n logits: ", logits)
        print("\n\n pred_class: ", pred_class)
        print("\n\n probabilities: ", probabilities)

        print("**************************************")
        st.divider()
        st.header("Classification Results" )
        if model_prediction == 0:
            pd_ = pd.DataFrame({
                        'Predicted': ["Negative"],
                        'Confidence': [probabilities[0]]
            })
            st.dataframe(pd_)
        else:
            pd_ = pd.DataFrame({
                        'Predicted': ["Positive"],
                        'Confidence': [probabilities[1]]
            })
            st.dataframe(pd_)
        
        print("**************************************")
        st.divider()
        st.header("Importance map (gradients)")

        model = model.model
        get_importance(signals, logits, pred_class, model, n_steps, n_features, ytrue)

#************************** Dashboard ***************************#
st.title("Chagas-XAI: An explainable deep learning tool for Chagas disease prediction using ECG signals")
st.divider()
my_logo = add_logo(width=width, height=height)
st.sidebar.image(my_logo)
st.sidebar.title("Artificial Intelligence in Biomedicine Group (ArBio)")
st.sidebar.link_button("Go to ArBio", "https://arbioiimas.github.io/ArBio/")
st.header("Load ECG signals")


if __name__ == "__main__":
    main()