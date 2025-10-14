import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
import webcolors

# ---------- COLOR DETECTION ----------
def get_dominant_colors(image, k=5):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (200, 200))
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = np.round(kmeans.cluster_centers_).astype(int)
    counts = np.bincount(kmeans.labels_)
    percentages = counts / counts.sum() * 100
    sorted_idx = np.argsort(-percentages)
    colors = colors[sorted_idx]
    percentages = percentages[sorted_idx]
    return colors, percentages


def closest_color(rgb):
    min_dist = float('inf')
    closest = None
    for name, hex_code in webcolors.CSS3_NAMES_TO_HEX.items():
        r, g, b = webcolors.hex_to_rgb(hex_code)
        dist = (r - rgb[0])**2 + (g - rgb[1])**2 + (b - rgb[2])**2
        if dist < min_dist:
            min_dist, closest = dist, name
    return closest


# ---------- STREAMLIT CONFIG ----------
st.set_page_config(page_title="Color Detective", page_icon="🎨", layout="centered")

# ---------- CUSTOM MODERN STYLES ----------
st.markdown("""
    <style>
        /* Animated gradient background */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(-45deg, #1a2a6c, #b21f1f, #fdbb2d, #283c86);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            color: #fff;
        }
        @keyframes gradientShift {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        /* Hide default Streamlit header and footer */
        header, footer {visibility: hidden;}

        /* App title */
        .title {
            text-align: center;
            font-size: 3em;
            font-weight: 800;
            letter-spacing: 1px;
            margin-bottom: 0.2em;
            background: -webkit-linear-gradient(45deg, #00f5d4, #f15bb5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .desc {
            text-align: center;
            color: #f0f0f0;
            font-size: 1.1em;
            margin-bottom: 2em;
        }

        /* Color card styling */
        .color-card {
            background: rgba(0, 0, 0, 0.45);
            backdrop-filter: blur(14px);
            border-radius: 14px;
            padding: 18px;
            margin: 20px 0;
            box-shadow: 0 4px 25px rgba(0,0,0,0.4);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .color-card:hover {
            transform: scale(1.03);
            box-shadow: 0 6px 30px rgba(255,255,255,0.25);
        }

        .bar-container {
            background-color: rgba(255,255,255,0.25);
            border-radius: 8px;
            height: 18px;
            margin-top: 10px;
        }

        .color-bar {
            height: 100%;
            border-radius: 8px;
            transition: width 0.8s ease-in-out;
        }

        /* File uploader label */
        .stFileUploader label {
            color: #fff;
            font-weight: 600;
        }

        /* Custom footer styling */
        footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 12px 0;
            font-size: 1em;
            color: #fff;
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(8px);
            border-top: 1px solid rgba(255,255,255,0.2);
            z-index: 999;
            opacity: 0;
            animation: fadeInFooter 2s ease forwards 1s;
        }
        footer:hover {
            background: rgba(0, 0, 0, 0.6);
            transition: 0.3s ease-in-out;
        }
        @keyframes fadeInFooter {
            to { opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

# ---------- STREAMLIT APP ----------
st.markdown("<div class='title'>🎨 Color Detective</div>", unsafe_allow_html=True)
st.markdown("<div class='desc'>Upload an image and explore its dominant colors in a stylish interface 🌈</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your image here 👇", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="📸 Uploaded Image", use_container_width=True)

    # Detect colors
    with st.spinner("Analyzing color palette... 🎨"):
        colors, percentages = get_dominant_colors(img, k=5)

    st.markdown("<h3 style='text-align:center; margin-top:30px; color:#fff;'>Dominant Colors</h3>", unsafe_allow_html=True)

    for color, perc in zip(colors, percentages):
        rgb = tuple(color)
        hex_code = '#%02x%02x%02x' % rgb
        try:
            name = webcolors.rgb_to_name(rgb)
        except ValueError:
            name = closest_color(rgb)

        st.markdown(f"""
            <div class='color-card'>
                <div style='display:flex; align-items:center;'>
                    <div style='width:65px; height:65px; background-color:{hex_code};
                                border-radius:10px; margin-right:15px;
                                border:2px solid rgba(255,255,255,0.7);'></div>
                    <div>
                        <div style='font-size:1.3em; font-weight:700;'>{name.title()}</div>
                        <div style='color:#ddd;'>RGB: {rgb} | Hex: {hex_code}</div>
                        <div style='color:#ddd;'>Dominance: {perc:.2f}%</div>
                    </div>
                </div>
                <div class='bar-container'>
                    <div class='color-bar' style='width:{perc}%; background-color:{hex_code};'></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ---------- CUSTOM FOOTER ----------
st.markdown("""
<footer>
Made with ❤️ by 
<span style="background: -webkit-linear-gradient(45deg, #00f5d4, #f15bb5);
             -webkit-background-clip: text;
             -webkit-text-fill-color: transparent;
             font-weight:bold;">Kishore</span>
 | Powered by Streamlit
</footer>
""", unsafe_allow_html=True)
