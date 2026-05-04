"""
Para correr: 
streamlit run iosa_app.py
"""
import streamlit as st
# import tkinter as tk
import os
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
# from tkinter import ttk
from scipy.ndimage import binary_closing, binary_fill_holes,binary_erosion,label,generate_binary_structure,iterate_structure
from scipy.spatial import ConvexHull
from skimage.measure import regionprops, find_contours
from streamlit_drawable_canvas import st_canvas

#****************************************************************#
# resize = 512
width = 100
height = 100

# Inicializaciones
if 'segmentation' not in st.session_state:
    st.session_state.segmentation = None

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


def loading_image():
    print("dentro de loading image")
    file_uploaded = st.file_uploader("Ototlith Image", type=["png","jpg","jpeg"])
    print(file_uploaded)
    if file_uploaded is not None:
        st.image(file_uploaded)
    else:
        print("dentro de else...")


def reset_state():
    st.session_state.automatic = 0
    st.rerun() 

def processing_automatic_segmentation():
    image_np = st.session_state["imagen_cargada"]   # recuperar

    gris = (image_np[:,:,0]).astype(np.uint8)

    # Umbralizar imagen -  ROI
    limit = st.slider("Threshold value", min_value=0, max_value=255, value=80)
    st.write(f"Threshold: {limit} gray level")

    # ── Buscar píxeles superiores al umbral ───────────────────
    mask        = gris > limit             # True donde el píxel > umbral
    mask_vis    = mask.astype(np.double)
    n_pixeles   = int(np.sum(mask))     # total de píxeles encontrados
    mask_uint8 = (mask * 255).astype(np.uint8)

    #####################################################
    # Mathematical morphology
    # ── strel('disk',5) + imclose ─────────────────────────────────
    disco  = generate_binary_structure(2, 2)
    se = iterate_structure(disco, 3)
    oto_close = binary_closing(mask, structure=se)
    oto_close_vis = oto_close.astype(np.double)

    # ── bwlabel(oto_close >= limit) ──────────────────────────────────
    L, n = label(oto_close)
    matri = np.array([np.sum(L == i) for i in range(1, n + 1)])
    Y = np.argmax(matri) + 1                 # índice región más grande
    oto_new = (L == Y)                       # máscara región más grande
    oto_seg = binary_fill_holes(oto_new)     # rellenar huecos internos
    oto_seg_vis = oto_seg.astype(np.double)

    # ── Guardar en session_state ──────────────────────────────────
    st.session_state["oto_seg"] = oto_seg

    #####################################################
    # Desplegar imagen otolito
    col1, col2 = st.columns(2)
    with col1:
        st.image(gris, caption="Otolith image", use_container_width=True)
    with col2:
        st.image(oto_seg_vis, caption="ROI Binary Mask",  use_container_width=True)

    ####################################################
    # Obtener el contorno de la ROI
    oto_per   = oto_seg & ~binary_erosion(oto_seg)
    C1, C2    = np.where(oto_per)
    
    # ── Propiedades región ────────────────────────────────────
    rows, cols  = np.where(oto_seg)
    area        = int(np.sum(oto_seg))
    centroid_y  = float(np.mean(rows))
    centroid_x  = float(np.mean(cols))
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    ############# Marcación de ejes  #############
    st.subheader("Draw Axes — Manual Input")

    # ── Línea 1 ───────────────────────────────────────────────────
    st.write("**Major Axis Coordinates:** Enter x1, y1, x2, y2")
    col1, col2, col3, col4 = st.columns(4)
    with col1: x1_l1 = st.number_input("x1", value=0.0, key="x1_l1")
    with col2: y1_l1 = st.number_input("y1", value=0.0, key="y1_l1")
    with col3: x2_l1 = st.number_input("x2", value=1.0, key="x2_l1")
    with col4: y2_l1 = st.number_input("y2", value=1.0, key="y2_l1")

    pos1 = np.array([[x1_l1, y1_l1],[x2_l1, y2_l1]])


    # ── Línea 2 ───────────────────────────────────────────────────
    st.write("**Minor Axis Coordinates:** Enter x1, y1, x2, y2")
    col1, col2, col3, col4 = st.columns(4)
    with col1: x1_l2 = st.number_input("x1", value=0.0, key="x1_l2")
    with col2: y1_l2 = st.number_input("y1", value=0.0, key="y1_l2")
    with col3: x2_l2 = st.number_input("x2", value=1.0, key="x2_l2")
    with col4: y2_l2 = st.number_input("y2", value=1.0, key="y2_l2")

    pos2 = np.array([[x1_l2, y1_l2],
                 [x2_l2, y2_l2]])

    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np)
    ax.set_title("Otolith image segmentation & Centroid")
    ax.axis("off")

    ax.plot(C2, C1, '.y', markersize=1)
    ax.plot(centroid_x, centroid_y, '*b', markersize=8)
    ax.plot(pos1[:,0], pos1[:,1], 'b-',  linewidth=2)
    ax.plot(pos1[:,0], pos1[:,1], 'bx',  markersize=8)
    ax.plot(pos2[:,0], pos2[:,1], 'r-',  linewidth=2)
    ax.plot(pos2[:,0], pos2[:,1], 'rx',  markersize=8)
    ax.add_patch(plt.Rectangle(
        (min_col, min_row),
        max_col - min_col,
        max_row - min_row,
        linewidth=1.5, edgecolor='red', facecolor='none'))
    ax.axis("off")
   
    st.pyplot(fig)
    plt.close(fig)


    #############################################################
    # Classis morphometric descriptors
    #############################################################
    # ── Inicializar session_state ─────────────────────────────────
    defaults = {
        "imagen_cargada": None,
        "oto_seg":        None,
        "results_df":     pd.DataFrame(),
        "dis1":           0.0,
        "dis2":           0.0,
        "selecc":         "Otolith",
        "pix":            1.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ── Validaciones (equivalente a isempty de MATLAB) ────────────
    if st.session_state["oto_seg"] is None:
        st.error("Segment the image first!")
        st.stop()
    
    # ── Calcular distancias ───────────────────────────────────────
    #if st.button("✅ Calculate Axes"):
    dis1 = np.sqrt((pos1[1,0]-pos1[0,0])**2 + (pos1[1,1]-pos1[0,1])**2)
    dis2 = np.sqrt((pos2[1,0]-pos2[0,0])**2 + (pos2[1,1]-pos2[0,1])**2)

    st.session_state["dis1"] = dis1
    st.session_state["dis2"] = dis2
    st.session_state["pos1"] = pos1
    st.session_state["pos2"] = pos2

    ###################################################
    # Pixel resolution
    pix = st.number_input(
        "Pixel Resolution (mm)",
    min_value = 0.0,
    value     = 1.0,       # valor por defecto
    step      = 0.001,
    format    = "%.4f"
    )

    # ── Guardar en session_state ──────────────────────────────────
    st.session_state["pix"] = pix

    if st.button("✅ Morphometric analysis"):

        pix    = st.session_state["pix"]    if st.session_state["pix"]   else 1.0
        dis1   = st.session_state["dis1"]   if st.session_state["dis1"]  else 0.0
        dis2   = st.session_state["dis2"]   if st.session_state["dis2"]  else 0.0
        selecc = st.session_state["selecc"] if st.session_state["selecc"] else "Otolith"

        oto_seg = st.session_state["oto_seg"]

        # ── regionprops (equivalente a MATLAB) ────────────────────────
        labeled = label(oto_seg)[0]
        props   = regionprops(labeled)[0]

        # ── Propiedades básicas ───────────────────────────────────────
        Area_mm2         = props.area             * pix * pix
        Perimeter_mm     = props.perimeter        * pix
        Circularity      = (4 * np.pi * props.area) / (props.perimeter ** 2) if props.perimeter > 0 else 0
        Convex_area_mm2  = props.convex_area      * pix * pix
        Eccentricity     = props.eccentricity
        Major_mm         = props.major_axis_length * pix
        Minor_mm         = props.minor_axis_length * pix

        # ── Métricas derivadas ────────────────────────────────────────
        Rectangularity   = Area_mm2 / (Major_mm * Minor_mm) if Major_mm * Minor_mm > 0 else 0
        Aspect_ratio     = (Minor_mm / Major_mm * 100)       if Major_mm > 0 else 0

        # ── Bounding box ──────────────────────────────────────────────
        bbox             = props.bbox              # (min_row, min_col, max_row, max_col)
        box_h            = (bbox[2] - bbox[0]) * pix
        box_w            = (bbox[3] - bbox[1]) * pix
        Box_area_mm2     = box_h * box_w
        Box_perimeter_mm = 2 * (box_h + box_w)

        # ── Equivalent diameter ───────────────────────────────────────
        Equivalent_diameter_mm = props.equivalent_diameter * pix

        # ── Feret diameters ───────────────────────────────────────────
        try:
            contours  = find_contours(oto_seg.astype(float), 0.5)
            hull      = ConvexHull(contours[0])
            hull_pts  = contours[0][hull.vertices]
            max_feret = 0
            min_feret = float("inf")
            for i in range(len(hull_pts)):
                for j in range(i + 1, len(hull_pts)):
                    d = np.linalg.norm(hull_pts[i] - hull_pts[j])
                    if d > max_feret: max_feret = d
                    if d < min_feret: min_feret = d
            Max_feret_mm = max_feret * pix
            Min_feret_mm = min_feret * pix
        except Exception:
            Max_feret_mm = Min_feret_mm = 0.0

        # ── Ejes manuales ────────────────────────────────────────────
        Manual_Axis1_mm = dis1 * pix
        Manual_Axis2_mm = dis2 * pix

        def compacidad2D(Otolito, Iedge):
            # Encontrar número de pixeles de la región
            ind_region = np.where(Otolito > 0)[0]
            n_pixeles = len(ind_region)
            area = n_pixeles

            # Encontrar perímetro de contacto
            pc = 0  # Áreas de contacto

            for i in range(area - 1):
                if ind_region[i] + 1 == ind_region[i + 1]:
                    pc += 1

            A = Otolito.T
            n_new = np.where(A)
            numvox = len(n_new[0])

            for i in range(numvox - 1):
                if n_new[0][i] + 1 == n_new[0][i + 1]:
                    pc += 1

            # Perímetro envolvente
            pe = (4 * n_pixeles) - (2 * pc)

            # Compacidad discreta
            Cd = (n_pixeles - (pe / 4)) / (n_pixeles - np.sqrt(n_pixeles))

            return Cd, pe, area, pc
        
        # ── Llamar y mostrar en Streamlit ─────────────────────────────
        oto_seg = st.session_state["oto_seg"]
        pix     = st.session_state.get("pix", 1.0)

        # Calcular
        Cd, pe, area, pc = compacidad2D(oto_seg, None)
        Cd = Cd * 1000

        Evolving_perimeter = pe * pix
        Contact_perimeter  = pc * pix

        ## Mostrar resultados
        #st.subheader("Compactness Results")
        #col1, col2, col3 = st.columns(3)
        #col1.metric("Discrete Compactness",  f"{Cd:.6f}")
        #col2.metric("Evolving Perimeter",    f"{Evolving_perimeter:.4f}")
        #col3.metric("Contact Perimeter",     f"{Contact_perimeter:.4f}")
            
        # Guardar en session_state
        st.session_state["Cd"]                 = Cd
        st.session_state["Evolving_perimeter"] = Evolving_perimeter
        st.session_state["Contact_perimeter"]  = Contact_perimeter

        # ── Mostrar métricas en Streamlit ─────────────────────────────
        st.subheader("Morphometric Results")

        # ── Guardar resultados en DataFrame ───────────────────────────
        row = {
            "Structure":              selecc,
            "Resolution_mm":          pix,
            "Area_mm2":               Area_mm2,
            "Perimeter_mm":           Perimeter_mm,
            "Circularity":            Circularity,
            "Major_Axis_mm":          Major_mm,
            "Minor_Axis_mm":          Minor_mm,
            "Rectangularity":         Rectangularity,
            "Aspect_ratio":           Aspect_ratio,
            "Box_area_mm2":           Box_area_mm2,
            "Box_perimeter_mm":       Box_perimeter_mm,
            "Convex_area_mm2":        Convex_area_mm2,
            "Eccentricity":           Eccentricity,
            "Equivalent_diameter_mm": Equivalent_diameter_mm,
            "Max_feret_mm":           Max_feret_mm,
            "Min_feret_mm":           Min_feret_mm,
            "Manual_Axis1_mm":        Manual_Axis1_mm,
            "Manual_Axis2_mm":        Manual_Axis2_mm,
            "Discrete Compactness":   Cd,
            "Evolving Perimeter":     Evolving_perimeter,
            "Contact Perimeter":      Contact_perimeter,
            }

        new_row = pd.DataFrame([row])
        st.session_state["results_df"] = pd.concat(
            [st.session_state["results_df"], new_row], ignore_index=True)
        

        # ── Mostrar tabla y descarga ───────────────────────────────────
        st.dataframe(st.session_state["results_df"])

        csv = st.session_state["results_df"].to_csv(index=False)
        st.download_button(
            label     = "📥 Download CSV",
            data      = csv,
            file_name = "Morphometric_Results.csv",
            mime      = "text/csv"
        )

    
def main():
    col1, col2, col3 = st.columns(3)

    with col1:
        print("---------------------")
        if st.button("Automatic",width="stretch"):
            st.session_state.segmentation = "automatic"
    with col2:
        print("---------------------")
        if st.button("Manual",width="stretch"):
            st.session_state.segmentation= "manual"
    with col3:
        print("---------------------")
        if st.button("Semi-Manual",width="stretch"):
            st.session_state.segmentation = "semimanual"


    if st.session_state.segmentation == "automatic":
        print("hola soy automatic")
        file_uploaded = st.file_uploader("Ototlith Image", type=["png","jpg","jpeg"])
        print(file_uploaded)

        if file_uploaded is not None:
            image = Image.open(file_uploaded).convert("RGB")
            image_np = np.array(image)
            print("image_np: ", image_np.shape)
            st.session_state["imagen_cargada"] = image_np   # guardar
            
            processing_automatic_segmentation()
        else:
            print("nada cargado")

    if st.session_state.segmentation == "manual":
        print("hola soy manual")
        file_uploaded = st.file_uploader("Ototlith Image", type=["png","jpg","jpeg"])
        print(file_uploaded)

    if st.session_state.segmentation == "semimanual":
        print("hola soy semimanual")
        file_uploaded = st.file_uploader("Ototlith Image", type=["png","jpg","jpeg"])
        print(file_uploaded)
    
   
#************************** Dashboard ***************************#
st.title("Invariant Otolith Shape Analysis IOSA Tool")
st.divider()
my_logo = add_logo(width=width, height=height)
st.sidebar.image(my_logo)
st.sidebar.title("Artificial Intelligence in Biomedicine Group (ArBio)")
st.sidebar.link_button("Go to ArBio", "https://arbioiimas.github.io/ArBio/")
st.header("Choose an option...")
# st.subheader("Preferably upload a histological image of cardiac tissue with hematoxylin and eosin staining at 40X.")

if __name__ == "__main__":
    main()