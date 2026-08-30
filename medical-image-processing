import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "brain_tumor_model.keras"
    )

    return model


model = load_model()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🧠 Brain Tumor Detection System")

st.write(
    """
    Upload a brain MRI image and the trained
    CNN model will predict whether the image
    belongs to the tumor or no-tumor class.
    """
)

st.warning(
    """
    ⚠️ This is an educational/research project.
    It is not a medical diagnostic tool.
    Do not use the prediction for medical decisions.
    """
)


# --------------------------------------------------
# UPLOAD IMAGE
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Display image
    st.subheader("📷 Uploaded MRI Image")

    st.image(
        image,
        caption="Brain MRI",
        use_container_width=True
    )


    # --------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------

    if st.button(
        "🔍 Analyze MRI",
        use_container_width=True
    ):

        # Convert image to NumPy
        img = np.array(image)

        # Resize to model input size
        img = cv2.resize(
            img,
            (224, 224)
        )

        # Normalize pixels
        img = img.astype(
            "float32"
        ) / 255.0

        # Add batch dimension
        img = np.expand_dims(
            img,
            axis=0
        )

        # --------------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------------

        prediction = model.predict(
            img,
            verbose=0
        )[0][0]


        # --------------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "🤖 Prediction Result"
        )


        if prediction >= 0.5:

            result = "Tumor Detected"

            confidence = prediction * 100

            st.error(
                f"⚠️ {result}"
            )

        else:

            result = "No Tumor Detected"

            confidence = (
                1 - prediction
            ) * 100

            st.success(
                f"✅ {result}"
            )


        # --------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        # --------------------------------------------------
        # PROBABILITY BAR
        # --------------------------------------------------

        st.subheader(
            "📊 Tumor Probability"
        )

        st.progress(
            float(prediction)
        )

        st.write(
            f"Tumor probability: "
            f"{prediction * 100:.2f}%"
        )


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("📚 About This Project")

st.write(
    """
    This project uses a Convolutional Neural Network
    (CNN) trained on brain MRI images.

    The model performs binary image classification:

    • Class 0 → No Tumor
    • Class 1 → Tumor

    Image preprocessing:

    • Resize image to 224 × 224
    • Convert pixel values to 0–1
    • Pass image to trained CNN
    """
)


st.subheader("🛠️ Technologies Used")

st.write(
    """
    Python | TensorFlow | Keras | OpenCV |
    NumPy | Pillow | Streamlit
    """
)
