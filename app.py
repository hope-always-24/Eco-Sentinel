
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Eco-Sentinel",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #edf5e9, #dcebd8);
    color: #183d2b;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}
.hero {
    background: linear-gradient(120deg, #123d2a, #286b45, #5b8f55);
    padding: 38px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 6px 18px rgba(20, 60, 35, 0.18);
}
.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 8px;
}
.hero p {
    color: #e0f0df;
    font-size: 17px;
}
.section-card {
    background: rgba(255,255,255,0.85);
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #c8ddc6;
    margin-bottom: 20px;
}
div[data-testid="stMetric"] {
    background: #f4f9f1;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #c8ddc6;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #183d2b !important;
}
.stButton > button {
    background: #286b45;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 25px;
    font-weight: bold;
}
.stButton > button:hover {
    background: #174b31;
    color: white;
    border: none;
}
h1, h2, h3 {
    color: #174b31;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("forestwatch_model.keras")
    with open("class_names.json", "r") as file:
        class_names = json.load(file)
    return model, class_names


try:
    model, class_names = load_model()
except Exception as error:
    st.error("Unable to load the AI model or class names file.")
    st.code(str(error))
    st.stop()


# HEADER
st.markdown("""
<div class="hero">
    <h1>🌿 Eco-Sentinel</h1>
    <p>AI-Powered Forest Monitoring System</p>
    <p>Supporting wildlife observation through artificial intelligence.</p>
</div>
""", unsafe_allow_html=True)


# INTRODUCTION
st.markdown("""
<div class="section-card">
    <h3>🌱 Welcome to Eco-Sentinel</h3>
    Upload an image to let the AI model classify it into one of
    four categories: Animals, Humans, Vehicles or Other.
</div>
""", unsafe_allow_html=True)


# IMAGE UPLOAD
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 📷 Upload Forest Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )
        except Exception:
            st.error("Please upload a valid JPG or PNG image.")
            st.stop()


with right:
    st.markdown("### 🔍 Image Analysis")

    st.markdown("""
    <div class="section-card">
        <h4>How it works</h4>
        <p>1. Upload a forest image.</p>
        <p>2. Click the Analyse Image button.</p>
        <p>3. The AI model examines the image.</p>
        <p>4. View the predicted category and scores.</p>
    </div>
    """, unsafe_allow_html=True)

    analyse_button = st.button(
        "🌲 Analyse Image",
        disabled=uploaded_file is None,
        use_container_width=True
    )


# PREDICTION
if analyse_button and uploaded_file is not None:

    try:
        with st.spinner("🌿 Analysing image..."):

            resized_image = image.resize((224, 224))
            image_array = np.array(resized_image)
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)

            predicted_index = int(np.argmax(predictions[0]))
            predicted_class = class_names[predicted_index]

            confidence = float(
                predictions[0][predicted_index]
            ) * 100

        st.divider()
        st.markdown("## 🌱 Analysis Results")

        result_col, score_col = st.columns(2)

        with result_col:
            st.markdown(
                f"""
                <div class="section-card">
                    <h3>Predicted Category</h3>
                    <h1>{predicted_class}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        with score_col:
            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

        st.markdown("### 📊 Category-wise Scores")

        for i, category in enumerate(class_names):
            score = float(predictions[0][i]) * 100
            st.write(f"**{category}:** {score:.2f}%")

        st.markdown("## 🚨 Monitoring Alerts")

        if confidence < 60:
            st.warning(
                "Low-confidence prediction. Manual verification is required."
            )
        elif predicted_class == "Humans":
            st.warning(
                "Human presence predicted. Please verify the image manually."
            )
        elif predicted_class == "Vehicles":
            st.warning(
                "Vehicle presence predicted. Please verify the image manually."
            )
        elif predicted_class == "Animals":
            st.success("Animal category predicted.")
        else:
            st.info("Image classified as Other.")

        st.caption(
            "AI predictions are estimates and are not proof of poaching, "
            "illegal activity, or the identity of any person."
        )

    except Exception as error:
        st.error("An error occurred during image analysis.")
        st.code(str(error))


# ABOUT
st.divider()

st.markdown("## 🌍 About Eco-Sentinel")

about1, about2, about3 = st.columns(3)

with about1:
    st.markdown("""
    <div class="section-card">
        <h4>🐾 Wildlife Monitoring</h4>
        Classifies uploaded images into four categories.
    </div>
    """, unsafe_allow_html=True)

with about2:
    st.markdown("""
    <div class="section-card">
        <h4>🤖 AI Technology</h4>
        Uses MobileNetV2 transfer learning for image classification.
    </div>
    """, unsafe_allow_html=True)

with about3:
    st.markdown("""
    <div class="section-card">
        <h4>🌳 Forest Awareness</h4>
        Demonstrates how AI can support forest observation.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="section-card">
    <h4>⚠️ Project Limitations</h4>
    This is an educational prototype trained on a limited image dataset.
    Its predictions may be incorrect and require human verification.
    It performs whole-image classification, not object detection.
</div>
""", unsafe_allow_html=True)


# FOOTER
st.markdown("""
<div style="text-align:center; padding:20px; color:#426b4d;">
    <b>🌿 Eco-Sentinel</b><br>
    Class XII AI Project | Protecting Nature Through Technology
</div>
""", unsafe_allow_html=True)
