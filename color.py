import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors

# ---------------------------------------------------------
# Safe fallback for different webcolors versions
# ---------------------------------------------------------
COLOR_MAP = (
    getattr(webcolors, "CSS3_NAMES_TO_HEX", None)
    or getattr(webcolors, "HTML4_NAMES_TO_HEX", None)
    or getattr(webcolors, "HTML3_NAMES_TO_HEX", None)
)

# ---------------------------------------------------------
# Helper: find closest named color
# ---------------------------------------------------------
def closest_color(rgb):
    if COLOR_MAP is None:
        return "Unknown"

    min_dist = float('inf')
    closest = None
    for name, hex_code in COLOR_MAP.items():
        r, g, b = webcolors.hex_to_rgb(hex_code)
        dist = (r - rgb[0]) ** 2 + (g - rgb[1]) ** 2 + (b - rgb[2]) ** 2
        if dist < min_dist:
            min_dist, closest = dist, name
    return closest

# ---------------------------------------------------------
# Extract dominant colors from image
# ---------------------------------------------------------
def get_dominant_colors(image, k=5):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="🎨 Color Detector", page_icon="🎨", layout="wide")

st.title("🎨 Image Color Detector")
st.markdown("Upload an image to find its dominant colors!")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

    st.write("Extracting colors... ⏳")
    colors = get_dominant_colors(image, k=5)

    st.subheader("🎯 Dominant Colors")
    cols = st.columns(len(colors))

    for i, color in enumerate(colors):
        rgb = tuple(color)
        hex_code = '#%02x%02x%02x' % rgb
        try:
            name = webcolors.rgb_to_name(rgb)
        except ValueError:
            name = closest_color(rgb)

        with cols[i]:
            st.markdown(
                f"""
                <div style='background-color:{hex_code};
                            border-radius:10px;
                            padding:40px;
                            text-align:center;
                            color:white'>
                    <b>{name}</b><br>
                    {hex_code}<br>
                    RGB: {rgb}
                </div>
                """,
                unsafe_allow_html=True
            )

else:
    st.info("📁 Please upload an image file to start.")

