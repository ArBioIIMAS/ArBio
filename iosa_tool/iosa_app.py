"""
Para correr: 
streamlit run iosa_app.py

streamlit                 1.57.0
streamlit-drawable-canvas 0.9.3

"""
import streamlit as st
import os
import numpy as np
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
from datetime import datetime

from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes,binary_erosion,label,generate_binary_structure,iterate_structure
from scipy.spatial import ConvexHull
from skimage.measure import regionprops, find_contours
from streamlit_drawable_canvas import st_canvas

#****************************************************************#
# resize = 512
width = 100
height = 100
#Inicializando ejes
ejeMayor_1x, ejeMayor_1y, ejeMayor_2x, ejeMayor_2y = -1, -1, -1, -1
ejeMenor_1x, ejeMenor_1y, ejeMenor_2x, ejeMenor_2y = -1, -1, -1, -1

# Inicializaciones
if 'segmentation' not in st.session_state:
    st.session_state.segmentation = None

#************************** Functions ***************************#
def load_logo():
    if not os.path.isfile('logo_arbio.png'):
        urllib.request.urlretrieve('https://arbioiimas.github.io/ArBio/images/logo_arbio.png', 'logo_arbio.png')
    return Image.open('logo_arbio.png')

def add_logo():
    """Read and return a resized logo"""
    logo = load_logo()
    modified_logo = logo#.resize((width, height))
    return modified_logo


def loading_image():
    print("dentro de loading image")
    file_uploaded = st.file_uploader("Ototlith Image", type=["png","jpg","jpeg"])
    print(file_uploaded)
    filename = file_uploaded.name
    if file_uploaded is not None:
        st.image(file_uploaded)
    else:
        print("dentro de else...")


def reset_state():
    st.session_state.automatic = 0
    st.rerun() 

def processing_manual(file_uploaded, filename):
    image_np = st.session_state["imagen_cargada"]   # recuperar
    gris = (image_np[:,:,0]).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np)
    ax.set_title("Otolith manual segmentation")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

    st.title("Free Pencil Drawing")
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=True,
        background_image= file_uploaded, #Image.open(file_uploaded).convert("RGB"),
        height=400,
        drawing_mode="freedraw", # This enables the free pencil
        key="canvas",
    )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])

        print("***************")
        print("objects: ", objects)
        for col in objects.select_dtypes(include=["object"]).columns:
            objects[col] = objects[col].astype("str")
        st.dataframe(objects)

        # oto_seg = st.session_state["oto_seg"]
        # # ── regionprops (equivalente a MATLAB) ────────────────────────
        # labeled = label(oto_seg)[0]
        # props   = regionprops(labeled)[0]
        # print("labeled: ", labeled)


def resize_with_ratio(img, new_height):
    # Calculate the ratio to maintain aspect ratio or scale specifically
    h_ratio = new_height / img.height
    new_width = int(img.width * h_ratio)
    
    # Resize the image using the calculated ratio
    return img.resize((new_width, new_height))

def processing_automatic_segmentation(file_uploaded, imagen_original):
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
    st.subheader("Draw Axes — Manual Input. \n" \
    "It is recommended to draw in the following order: major axis and then minor axis.")

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

    #Recuperando el tamaño original de la imagen
    img_width, img_height = gris.shape
    display_width = 400
    ratio = img_width / display_width #Razón
    height = int(img_height / ratio)
    image_coors = imagen_original
    small_img = image_coors.resize((display_width, int(img_height / ratio)))
    small_img_array = np.array(small_img)
    print("img_width: ", img_width, "img_height: ", img_height, "ratio: ", ratio, "height: ", height, "small_img_array: ", small_img_array.shape)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        background_color="#FFFFFF",
        update_streamlit=True,
        width=small_img_array.shape[0],
        height= small_img_array.shape[1],
        background_image = small_img, #imagen_original, #Image.open(file_uploaded).convert("RGB"),
        drawing_mode="point", #"freedraw", # This enables the free pencil
        key="canvas",
    )

    # Do something interesting with the image data and paths
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        
        for col in objects.select_dtypes(include=["object"]).columns:
            objects[col] = objects[col].astype("str")
        # st.dataframe(objects)

    try:
        left = objects["left"].to_list()
        top = objects["top"].to_list()

        # print("left: ", left, type(left))
        # print("top: ", top, type(top))

        ejeMayor_1x = int(left[-4] * ratio)
        ejeMayor_1y = int(top[-4] * ratio)

        ejeMayor_2x = int(left[-3] * ratio)
        ejeMayor_2y = int(top[-3] * ratio)

        # print("ejeMayor_1x: ", ejeMayor_1x, "ejeMayor_1y: ", ejeMayor_1y)
        # print("ejeMayor_2x: ", ejeMayor_2x, "ejeMayor_2y: ", ejeMayor_2y)

        ejeMenor_1x = int(left[-2] * ratio)
        ejeMenor_1y = int(top[-2] * ratio)

        ejeMenor_2x = int(left[-1] * ratio)
        ejeMenor_2y = int(top[-1] * ratio)

        # print("ejeMenor_1x: ", ejeMenor_1x, "ejeMenor_1y: ", ejeMenor_1y)
        # print("ejeMenor_2x: ", ejeMenor_2x, "ejeMenor_2y: ", ejeMenor_2y)
        
        # ── Línea 1 ───────────────────────────────────────────────────
        pos1 = np.array([[ejeMayor_1x, ejeMayor_1y],[ejeMayor_2x, ejeMayor_2y]])

        # ── Línea 2 ───────────────────────────────────────────────────
        pos2 = np.array([[ejeMenor_1x, ejeMenor_1y],[ejeMenor_2x, ejeMenor_2y]])

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
        ax.add_patch(plt.Rectangle((min_col, min_row),max_col - min_col,max_row - min_row,linewidth=1.5, edgecolor='red', facecolor='none'))
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

        # # ── Calcular distancias ───────────────────────────────────────
        #if st.button("✅ Calculate Axes"):
        dis1 = np.sqrt((pos1[1,0]-pos1[0,0])**2 + (pos1[1,1]-pos1[0,1])**2)
        dis2 = np.sqrt((pos2[1,0]-pos2[0,0])**2 + (pos2[1,1]-pos2[0,1])**2)

        print("holaaa", dis1, dis2)
        st.session_state["dis1"] = dis1
        st.session_state["dis2"] = dis2
        st.session_state["pos1"] = pos1
        st.session_state["pos2"] = pos2
        
    except:
        print("Continúa..")


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
    
    # ###################################################
    # # Pixel resolution
    pix = st.number_input("Pixel Resolution (mm)",
    min_value = 0.0,
    value     = 1.0,       # valor por defecto
    step      = 0.001,
    format    = "%.4f"
    )

    # # ── Guardar en session_state ──────────────────────────────────
    st.session_state["pix"] = pix

    if st.button("✅ Morphometric analysis"):
        pix    = st.session_state["pix"]    if st.session_state["pix"]   else 1.0
        dis1   = st.session_state["dis1"]   if st.session_state["dis1"]  else 0.0
        dis2   = st.session_state["dis2"]   if st.session_state["dis2"]  else 0.0
        selecc = st.session_state["selecc"] if st.session_state["selecc"] else "Otolith"
        oto_seg = st.session_state["oto_seg"]

    #     # ── regionprops (equivalente a MATLAB) ────────────────────────
        labeled = label(oto_seg)[0]
        props   = regionprops(labeled)[0]

    #     # ── Propiedades básicas ───────────────────────────────────────
        Area_mm2         = props.area             * pix * pix
        Perimeter_mm     = props.perimeter        * pix
        Circularity      = (4 * np.pi * props.area) / (props.perimeter ** 2) if props.perimeter > 0 else 0
        Convex_area_mm2  = props.convex_area      * pix * pix
        Eccentricity     = props.eccentricity
        Major_mm         = props.major_axis_length * pix
        Minor_mm         = props.minor_axis_length * pix

    #     # ── Métricas derivadas ────────────────────────────────────────
        Rectangularity   = Area_mm2 / (Major_mm * Minor_mm) if Major_mm * Minor_mm > 0 else 0
        Aspect_ratio     = (Minor_mm / Major_mm * 100)       if Major_mm > 0 else 0

    #     # ── Bounding box ──────────────────────────────────────────────
        bbox             = props.bbox              # (min_row, min_col, max_row, max_col)
        box_h            = (bbox[2] - bbox[0]) * pix
        box_w            = (bbox[3] - bbox[1]) * pix
        Box_area_mm2     = box_h * box_w
        Box_perimeter_mm = 2 * (box_h + box_w)

    #     # ── Equivalent diameter ───────────────────────────────────────
        Equivalent_diameter_mm = props.equivalent_diameter * pix

    #     # ── Feret diameters ───────────────────────────────────────────
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

    #     # ── Ejes manuales ────────────────────────────────────────────
        Manual_Axis1_mm = dis1 * pix
        Manual_Axis2_mm = dis2 * pix

        def compacidad2D(Otolito):
            # Encontrar número de pixeles de la región
            ind_region = np.flatnonzero(Otolito > 0)
            n_pixeles = len(ind_region)
            area = n_pixeles

            # Encontrar perímetro de contacto
            pc = 0  # Áreas de contacto

            for i in range(area - 1):
                if ind_region[i] + 1 == ind_region[i + 1]:
                    pc += 1

            A = Otolito.T
            n_new = np.flatnonzero(A)
            numvox = len(n_new)

            for i in range(numvox - 1):
                if n_new[i] + 1 == n_new[i + 1]:
                    pc += 1

            # Perímetro envolvente
            pe = (4 * n_pixeles) - (2 * pc)

            # Compacidad discreta
            Cd = (n_pixeles - (pe / 4)) / (n_pixeles - np.sqrt(n_pixeles))
            
            return Cd, pe, area, pc
        
    #     #######################################################
    #     # Angulo de contingencia
    #     # #######################################################     
        def _angulo_vector(y1, x1):
            """Calcula ángulo (0-360) de un vector — recibe escalares float."""
            y1 = float(y1); x1 = float(x1)
            if y1 == 0:
                return 0.0 if x1 > 0 else 180.0
            elif y1 > 0:
                if x1 == 0:  return 90.0
                elif x1 > 0: return np.degrees(np.arctan(y1 / x1))
                else:        return np.degrees(np.arctan(y1 / x1)) + 180.0
            else:
                if x1 == 0:  return 270.0
                elif x1 > 0: return np.degrees(np.arctan(y1 / x1)) + 360.0
                else:        return np.degrees(np.arctan(y1 / x1)) + 180.0

        def angulo_contingencia(y, x, p):
            # ── Aplanar a 1D para evitar errores de dimensión ─────────
            y = y.flatten().astype(float)
            x = x.flatten().astype(float)
            n = y.shape[0]

            if p == 0:
                if n < 3:
                    st.warning('Insufficient points for closed curve.')  
                    return np.array([])
                else:
                    yc = y[0:2].copy()
                    xc = x[0:2].copy()
                    y  = np.append(y, [yc[0], yc[1]])
                    x  = np.append(x, [xc[0], xc[1]])

                    alpha = np.zeros(n)
                    for i in range(n):
                        y1 = float(y[i+1] - y[i]);   x1 = float(x[i+1] - x[i])
                        y2 = float(y[i+2] - y[i+1]); x2 = float(x[i+2] - x[i+1])

                        tetha1 = _angulo_vector(y1, x1)
                        tetha2 = _angulo_vector(y2, x2)

                        ang = tetha2 - tetha1
                        if ang >  180: ang -= 360
                        if ang < -180: ang += 360

                        alpha[i] = ang / 180

            else:
                if n < 2:
                    st.warning('Insufficient points for open curve.')  # print → st.warning
                    return np.array([])
                else:
                    alpha = np.zeros(n - 2)
                    for i in range(n - 2):
                        y1 = float(y[i+1] - y[i]);   x1 = float(x[i+1] - x[i])
                        y2 = float(y[i+2] - y[i+1]); x2 = float(x[i+2] - x[i+1])

                        tetha1 = _angulo_vector(y1, x1)
                        tetha2 = _angulo_vector(y2, x2)

                        ang = tetha2 - tetha1
                        if ang >  180: ang -= 360
                        if ang < -180: ang += 360

                        alpha[i] = ang / 180

            return alpha.flatten()
        

        ##########################################################
        #### Calcular tortuosidad discreta y No-circularidad
        ####
        #########################################################
        def obtener_tortuosidad(matriz, i):
            # ── Extraer coordenadas ───────────────────────────────────
            ys = matriz[:, 0].flatten().astype(float)
            xs = matriz[:, 1].flatten().astype(float)

            # ── Calcular ángulos de contingencia ──────────────────────
            alpha   = angulo_contingencia(ys, xs, 0).flatten()
            alpha_T = -alpha[:i-1]

            if alpha_T.size == 0:
                st.warning("alpha_T está vacío — verifica el contorno.")
                return 0.0, 0.0

            # ── Tortuosidad normalizada ───────────────────────────────
            T   = float(np.sum(np.abs(alpha_T)))
            Tn  = T / alpha_T.size

            # ── No-circularidad normalizada ───────────────────────────
            Acc = float(np.sum(alpha_T))
            Am  = Acc / len(alpha_T)
            Dc  = float(np.sum(np.abs(alpha_T - Am)))
            DcN = Dc / alpha_T.size

            return float(Tn), float(DcN)
        

        # ── Llamar y mostrar en Streamlit ─────────────────────────────
        oto_seg = st.session_state["oto_seg"]
        pix     = st.session_state.get("pix", 1.0)

        # Calcular compacidad discreta
        Cd, pe, area, pc = compacidad2D(oto_seg)
        
        Evolving_perimeter = pe * pix
        Contact_perimeter  = pc * pix

        # Calcular tortuosidad discreta
        perim = find_contours(oto_seg.astype(float), level=0.5)
        rows = perim[0][:, 0]                    # filas
        cols = perim[0][:, 1]                    # columnas
        MATRIZ     = np.column_stack([rows, cols]).astype(float)
        i, j = MATRIZ.shape
        coord_T        = np.zeros((i + 1, j))
        coord_T[:i, :] = MATRIZ        # coord_T(1:i,:) = MATRIZ
        coord_T[i,  :] = MATRIZ[1, :]  # coord_T(i+1,:) = MATRIZ(2,:)
        Discrete_tortuosity, Normalized_non_circularity = obtener_tortuosidad(coord_T, i)


        # Guardar en session_state
        st.session_state["Cd"]                 = Cd
        st.session_state["Evolving_perimeter"] = Evolving_perimeter
        st.session_state["Contact_perimeter"]  = Contact_perimeter
        st.session_state["Discrete_Compactness"] = Discrete_tortuosity
        st.session_state["Normalized_non_circularity"] = Normalized_non_circularity


        # ── Mostrar métricas en Streamlit ─────────────────────────────
        st.subheader("Morphometric Results")

        # Get current time
        now = datetime.now()

        # ── Guardar resultados en DataFrame ───────────────────────────
        row = {
            "Filename":               file_uploaded.name,
            "Datetime":               now,
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
            "Manual_Axis1_mm":        Manual_Axis1_mm,
            "Manual_Axis2_mm":        Manual_Axis2_mm,
            "Discrete Compactness":   Cd,
            "Evolving Perimeter":     Evolving_perimeter,
            "Contact Perimeter":      Contact_perimeter,
            "Discrete Tortuosity":    Discrete_tortuosity,
            "Normalized Non-Circularity": Normalized_non_circularity
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
            mime      = "text/csv")
    
def main():
    col1, col2, col3 = st.columns(3)

    with col1:
        print("---------------------")
        if st.button("Automatic",): #width="stretch"
            st.session_state.segmentation = "automatic"
    # with col2:
    #     print("---------------------")
    #     if st.button("Manual"):
    #         st.session_state.segmentation= "manual"
    # with col3:
    #     print("---------------------")
    #     if st.button("Semi-Manual"):
    #         st.session_state.segmentation = "semimanual"


    if st.session_state.segmentation == "automatic":
        file_uploaded = st.file_uploader("Ototlith Image", type=["png","jpg","jpeg"])
        if file_uploaded is not None:
            imagen_original = Image.open(file_uploaded).convert("RGB")
            image_np = np.array(imagen_original)
            st.session_state["imagen_cargada"] = image_np   # guardar
            processing_automatic_segmentation(file_uploaded, imagen_original)
        else:
            print("nada cargado")

    if st.session_state.segmentation == "manual":
        file_uploaded = st.file_uploader("Otolith Image", type=["png","jpg","jpeg"])
        print(file_uploaded)
        if file_uploaded is not None:
            image = Image.open(file_uploaded).convert("RGB")
            image_np = np.array(image)
            st.session_state["imagen_cargada"] = image_np   # guardar
            processing_manual(file_uploaded, file_uploaded.name)
        else:
            print("nada cargado")

    if st.session_state.segmentation == "semimanual":
        file_uploaded = st.file_uploader("Otolith Image", type=["png","jpg","jpeg"])
        print(file_uploaded)
    
   
#************************** Dashboard ***************************#
st.title("MorphOtolith (MO-Tool): a tool for extracting discrete morphometric descriptors from otolith images")
st.divider()
my_logo = add_logo()
st.sidebar.image(my_logo)
st.sidebar.title("Artificial Intelligence in Biomedicine Group (ArBio)")
st.sidebar.link_button("Go to ArBio", "https://arbioiimas.github.io/ArBio/")
st.header("Choose an option...")
# st.subheader("Preferably upload a histological image of cardiac tissue with hematoxylin and eosin staining at 40X.")

if __name__ == "__main__":
    main()
