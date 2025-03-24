import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import gdown
import os

# Load the model (Google Drive method if needed)
MODEL_PATH = "inswapper_128.onnx"
# if not os.path.exists(MODEL_PATH):
#     url = "YOUR_GOOGLE_DRIVE_DIRECT_LINK"  # Replace with your link
#     gdown.download(url, MODEL_PATH, quiet=False)

swapper = get_model(MODEL_PATH)
swapper.prepare(ctx_id=0, det_size=(128, 128))

# Initialize face analysis
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

def swap_n_show(img1, img2):
    """ Swaps faces between two uploaded images and displays results """

    # Convert to OpenCV format
    img1 = cv2.imdecode(np.frombuffer(img1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2.read(), np.uint8), cv2.IMREAD_COLOR)

    # Detect faces
    face1 = app.get(img1)[0]
    face2 = app.get(img2)[0]

    # Perform face swapping
    img1_ = swapper.get(img1, face1, face2, paste_back=True)
    img2_ = swapper.get(img2, face2, face1, paste_back=True)

    # Show results
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cv2.cvtColor(img1_, cv2.COLOR_BGR2RGB))
    axs[0].axis("off")
    axs[1].imshow(cv2.cvtColor(img2_, cv2.COLOR_BGR2RGB))
    axs[1].axis("off")
    st.pyplot(fig)

# Streamlit UI
st.title("Face Swap App")
st.write("Upload two images to swap faces!")

img1 = st.file_uploader("Upload First Image", type=["jpg", "png", "jpeg"])
img2 = st.file_uploader("Upload Second Image", type=["jpg", "png", "jpeg"])

if img1 and img2:
    st.write("Processing...")
    swap_n_show(img1, img2)
