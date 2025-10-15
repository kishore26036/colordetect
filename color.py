import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
import webcolors

# ---------- SAFE COLOR NAME FUNCTION ----------
def closest_color(requested_color):
    """Finds the closest HTML color name for an RGB value, safe across versions."""
    try:
        color_maps = [
            getattr(webcolors, "CSS3_NAMES_TO_HEX", None),
            getattr(webcolors, "HTML4_NAMES_TO_HEX", None),
            getattr(webcolors, "HTML5_NAMES_TO_HEX", None),
        ]
        # Fallback for internal definitions (latest webcolors)
        if not any(color_maps):
            from webcolors._definitions import _CSS3_NAMES_TO_HEX as internal_map
            color_map = internal_map
        else:
            color_map = next((m for m in color_maps if m), None)
    except Exception:
        return "Unknown"

    if not color_map:
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

# ---------- STREAMLIT PAGE CONFIG ----------
st.set_page_config(page_title="Color Detective 🎨", page_icon="🎨", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Poppins', sans-serif;
}
h1 {
    text-align: center;
    color: #fff;
    text-shadow: 2px 2px 6px rgba(255,255,255,0.3);
}
.stApp {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
}
.color-card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
footer {
    text-align: center;
    margin-top: 30px;
    font-size: 14px;
    color: #bbb;
}
.upload-btn {
    background-color: #1f6feb !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- APP TITLE ----------
st.title("🎨 Color Detective")
st.write("Upload an image to analyze its dominant colors in style! 🌈")

# ---------- FILE UPLOADER ----------
uploaded_file = st.file_uploader("Choose an image", type=["png","jpg","jpeg"], label_visibility="collapsed")

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="📸 Uploaded Image", use_container_width=True)

    # Detect colors
    with st.spinner("🔍 Detecting dominant colors..."):
        colors, percentages = get_dominant_colors(img, k=5)

    st.subheader("🎨 Detected Color Palette")
    for i, color in enumerate(colors):
        rgb = tuple(color)
        hex_code = '#%02x%02x%02x' % rgb
        try:
            name = webcolors.rgb_to_name(rgb)
        except ValueError:
            name = closest_color(rgb)

        st.markdown(f"""
        <div class="color-card">
            <div style="display:flex; align-items:center;">
                <div style="width:60px; height:60px; background-color:{hex_code}; border-radius:10px; margin-right:15px;"></div>
                <div>
                    <b style="font-size:18px; color:#fff;">{name.title()}</b><br>
                    <span style="color:#ddd;">RGB: {rgb}</span><br>
                    <span style="color:#ddd;">HEX: {hex_code}</span><br>
                    <span style="color:#ddd;">Share: {percentages[i]:.2f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("<footer>Made with ❤️ by Kishore | Powered by Streamlit</footer>", unsafe_allow_html=True)
