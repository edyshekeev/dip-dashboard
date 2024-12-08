import streamlit as st
import numpy as np
from PIL import Image, ImageFilter

st.sidebar.success("Choose an algorithm")

# Handle Image Loading with Error Checking
try:
    image = Image.open("images.jpg")
    st.image(image, caption="Original Image", use_container_width=True)
except FileNotFoundError:
    st.error("Image file not found. Please make sure the 'images.png' file is available in the directory.")

# Sidebar mask options
mask = st.selectbox("Choose a mask (filter)", ("Gaussian Blur", "Box Blur", "Median", "Sharpening"))

# Apply the chosen filter using Pillow
if 'image' in locals():  # Only apply filters if the image was loaded successfully
    if mask == "Gaussian Blur":
        image_output = image.filter(ImageFilter.GaussianBlur(radius=1.5))
    elif mask == "Box Blur":
        image_output = image.filter(ImageFilter.BoxBlur(radius=1))
    elif mask == "Median":
        image_output = image.filter(ImageFilter.MedianFilter(size=3))
    elif mask == "Sharpening":
        image_output = image.filter(ImageFilter.SHARPEN)

    st.image(image_output, caption=f"Image with {mask}", use_container_width=True)
