import streamlit as st
import numpy as np
from PIL import Image, ImageOps

# Function to apply dilation
def dilation(image, kernel_size):
    h, w = image.shape
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode="constant", constant_values=0)
    dilated_image = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            dilated_image[i, j] = np.max(padded_image[i:i+kernel_size, j:j+kernel_size])
    
    return dilated_image

# Function to apply erosion
def erosion(image, kernel_size):
    h, w = image.shape
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode="constant", constant_values=255)
    eroded_image = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            eroded_image[i, j] = np.min(padded_image[i:i+kernel_size, j:j+kernel_size])
    
    return eroded_image

# Streamlit app
st.title("Image Morphology: Dilation and Erosion")

# Load the predefined image
image_path = "images.jpg"
try:
    img = Image.open(image_path)
    img = ImageOps.grayscale(img)  # Convert to grayscale for morphology operations
    img_array = np.array(img)

    st.image(img, caption="Original Image", use_conatiner_width=True)

    # Select morphology operation
    morph_type = st.selectbox("Choose morphology operation", ["Dilation", "Erosion"])

    # Select kernel size
    kernel_size = st.slider("Select kernel size", min_value=3, max_value=11, step=2, value=3)

    if st.button("Apply Morphology"):
        if morph_type == "Dilation":
            result_image = dilation(img_array, kernel_size)
            st.image(Image.fromarray(result_image), caption=f"Dilated Image (Kernel Size: {kernel_size}x{kernel_size})", use_container_width=True)
        elif morph_type == "Erosion":
            result_image = erosion(img_array, kernel_size)
            st.image(Image.fromarray(result_image), caption=f"Eroded Image (Kernel Size: {kernel_size}x{kernel_size})", use_container_width=True)

except FileNotFoundError:
    st.error(f"Image file '{image_path}' not found! Please ensure the image is in the correct path.")
