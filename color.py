import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors

# -----------------------------
# Helper functions
# -----------------------------
def closest_color(requested_color):
    """Finds the closest HTML color name for an RGB value."""
    try:
        # Use available color maps (works with any webcolors version)
        color_map = getattr(webcolors, "CSS3_NAMES_TO_HEX", None) \
                 or getattr(webcolors, "HTML4_NAMES_TO_HEX", None) \
                 or getattr(webcolors, "HTML3_NAMES_TO_HEX", None)
    except Exception:
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

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
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
st.set_page_config(page_title="🎨 Color Detective", page_icon="🎨", layout="wide")

# Gradient animated background + clean UI styling
st.markdown(
    """
    <style>
    /* Gradient animation */
    .stApp {
        background: linear-gradient(120deg, #1a2a6c, #b21f1f, #fdbb2d, #283c86);
        background-size: 400% 400%;
        animation: gradientFlow 15s ease infinite;
        color: #fff;
    }
    @keyframes gradientFlow {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* Title and description */
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

    /* File uploader styling */
    .stFileUploader label {
        background: rgba(0,0,0,0.6);
        color: #fff !important;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        border: 2px solid rgba(255,255,255,0.5);
    }
    .stFileUploader:hover label {
        background: rgba(255,255,255,0.2);
        cursor: pointer;
        transition: 0.3s;
    }

    /* Color card design */
    .color-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        margin: 15px;
        box-shadow: 0 4px 25px rgba(0,0,0,0.3);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .color-card:hover {
        transform: scale(1.03);
        box-shadow: 0 8px 30px rgba(255,255,255,0.25);
    }
    .color-box {
        width: 100%;
        height: 100px;
        border-radius: 10px;
        border: 2px solid rgba(255,255,255,0.5);
        margin-bottom: 15px;
    }
    .color-info {
        font-size: 1.1em;
        color: #fff;
        line-height: 1.8em;
    }

    footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: #fff;
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        padding: 8px 0;
        font-size: 1em;
        border-top: 1px solid rgba(255,255,255,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App header
st.markdown("<div class='title'>🎨 Color Detective</div>", unsafe_allow_html=True)
st.markdown("<div class='desc'>Upload an image and explore its dominant colors in a beautiful format 🌈</div>", unsafe_allow_html=True)

# Upload section
uploaded_file = st.file_uploader("📸 Upload your image below", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read and show image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

    with st.spinner("🎨 Detecting colors... Please wait!"):
        colors, percentages = get_dominant_colors(img, k=5)

    st.markdown("<h3 style='text-align:center;margin-top:30px;'>Top Dominant Colors</h3>", unsafe_allow_html=True)

    cols = st.columns(len(colors))
    for i, (color, perc) in enumerate(zip(colors, percentages)):
        rgb = tuple(color)
        hex_code = '#%02x%02x%02x' % rgb
        try:
            # Try to get exact name first
            name = webcolors.rgb_to_name(rgb)
        except ValueError:
            # Fall back to nearest color name
            name = closest_color(rgb)

        # Capitalize name properly
        name = name.replace("-", " ").title() if name else "Unknown"

        with cols[i]:
            st.markdown(
                f"""
                <div class="color-card">
                    <div class="color-box" style="background-color:{hex_code};"></div>
                    <div class="color-info">
                        <b>Name:</b> {name}<br>
                        <b>HEX:</b> {hex_code}<br>
                        <b>RGB:</b> {rgb}<br>
                        <b>Dominance:</b> {perc:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Footer
st.markdown(
    """
<footer>
Made with ❤️ by <b style="background: -webkit-linear-gradient(45deg, #00f5d4, #f15bb5);
             -webkit-background-clip: text;
             -webkit-text-fill-color: transparent;">Kishore</b> | Powered by Streamlit
</footer>
""",
    unsafe_allow_html=True,
)
