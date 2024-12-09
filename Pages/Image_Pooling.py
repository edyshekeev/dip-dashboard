import streamlit as st
import numpy as np
from PIL import Image

# Function to apply max pooling
def max_pooling(image, kernel_size):
    h, w, c = image.shape
    new_h = h - h % kernel_size  # Adjust height to fit kernel size
    new_w = w - w % kernel_size  # Adjust width to fit kernel size
    image = image[:new_h, :new_w, :]  # Crop the image to match dimensions

    pooled_image = np.zeros((new_h // kernel_size, new_w // kernel_size, c), dtype=np.uint8)

    for k in range(c):  # Process each channel separately
        for i in range(0, new_h, kernel_size):
            for j in range(0, new_w, kernel_size):
                pooled_image[i // kernel_size, j // kernel_size, k] = np.max(
                    image[i:i + kernel_size, j:j + kernel_size, k]
                )
    return pooled_image

# Function to apply average pooling
def average_pooling(image, kernel_size):
    h, w, c = image.shape
    new_h = h - h % kernel_size  # Adjust height to fit kernel size
    new_w = w - w % kernel_size  # Adjust width to fit kernel size
    image = image[:new_h, :new_w, :]  # Crop the image to match dimensions

    pooled_image = np.zeros((new_h // kernel_size, new_w // kernel_size, c), dtype=np.uint8)

    for k in range(c):  # Process each channel separately
        for i in range(0, new_h, kernel_size):
            for j in range(0, new_w, kernel_size):
                pooled_image[i // kernel_size, j // kernel_size, k] = np.mean(
                    image[i:i + kernel_size, j:j + kernel_size, k]
                )
    return pooled_image

# Streamlit app
st.title("Image Pooling (Max and Average)")

# Load the predefined image
image_path = "images.jpg"
try:
    img = Image.open(image_path)
    img_array = np.array(img)

    st.image(img, caption="Original Image", use_container_width=True)

    # Select pooling type
    pooling_type = st.selectbox("Choose pooling type", ["Max Pooling", "Average Pooling"])

    # Select kernel size
    kernel_size = st.slider("Select kernel size", min_value=2, max_value=10, step=1, value=2)

    if st.button("Apply Pooling"):
        if pooling_type == "Max Pooling":
            pooled_img = max_pooling(img_array, kernel_size)
            st.image(Image.fromarray(pooled_img), caption=f"Max Pooled Image (Kernel Size: {kernel_size}x{kernel_size})", use_container_width=True)
        elif pooling_type == "Average Pooling":
            pooled_img = average_pooling(img_array, kernel_size)
            st.image(Image.fromarray(pooled_img), caption=f"Average Pooled Image (Kernel Size: {kernel_size}x{kernel_size})", use_container_width=True)

except FileNotFoundError:
    st.error(f"Image file '{image_path}' not found! Please ensure the image is in the correct path.")
