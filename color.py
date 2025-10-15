import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors

# -----------------------------
# Helper functions
# -----------------------------

def closest_color(requested_color):
    """Find closest CSS3 color name for an RGB value."""
    try:
        # Try modern version of webcolors
        color_map = getattr(webcolors, "CSS3_NAMES_TO_HEX", getattr(webcolors, "HTML4_NAMES_TO_HEX", None))
        if not color_map:
            raise AttributeError
    except AttributeError:
        return "Unknown"

    min_dist = float("inf")
    closest_name = "Unknown"
    for name, hex_code in color_map.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
        dist = (r_c - requested_color[0]) ** 2 + (g_c - requested_color[1]) ** 2 + (b_c - requested_color[2]) ** 2
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name


def get_dominant_colors(image, k=5):
    """Extract k dominant colors using KMeans clustering."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_data = img_rgb.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(img_data)

    colors = kmeans.cluster_centers_.astype(int)
    labels = np.bincount(kmeans.labels_)
    percentages = (labels / len(kmeans.labels_)) * 100

    sorted_idx = np.argsort(percentages)[::-1]
    colors = colors[sorted_idx]
    percentages = percentages[sorted_idx]

    return colors, percentages


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🎨 AI Color Detector", page_icon="🎨", layout="wide")

# Gradient background
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #2b1055, #7597de, #1dd1a1);
        background-attachment: fixed;
        color: white;
    }
    .stApp {
        background: linear-gradient(120deg, #020024, #090979, #00d4ff);
        background-size: 400% 400%;
        animation: gradientFlow 10s ease infinite;
        color: white;
    }
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .color-card {
        border-radius: 20px;
        padding: 25px;
        margin: 10px;
        box-shadow: 0px 0px 25px rgba(255,255,255,0.2);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .color-card:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 40px rgba(255,255,255,0.4);
    }
    footer {
        text-align: center;
        margin-top: 40px;
        color: #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1 style='text-align:center;'>🎨 AI Color Detector</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Upload an image to extract its dominant colors</h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📁 Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display image
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Detecting dominant colors..."):
        colors, percentages = get_dominant_colors(image, k=5)

    st.subheader("🎯 Detected Dominant Colors")

    cols = st.columns(len(colors))
    for i, (color, percent) in enumerate(zip(colors, percentages)):
        rgb = tuple(color)
        hex_code = '#%02x%02x%02x' % rgb

        try:
            name = webcolors.rgb_to_name(rgb)
        except ValueError:
            name = closest_color(rgb)

        with cols[i]:
            st.markdown(
                f"""
                <div class='color-card' style='background-color:{hex_code}; color:white;'>
                    <b>{name.title()}</b><br>
                    HEX: {hex_code}<br>
                    RGB: {rgb}<br>
                    <b>{percent:.2f}%</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Footer
st.markdown("<footer>Made with ❤️ by Kishore | Powered by Streamlit</footer>", unsafe_allow_html=True)
