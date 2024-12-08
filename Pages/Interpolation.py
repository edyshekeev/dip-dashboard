import streamlit as st
import numpy as np
from PIL import Image

# Nearest Neighbor Interpolation
def nearest_neighbor_interpolation(image, new_width, new_height):
    original_width, original_height = image.size
    new_image = Image.new(image.mode, (new_width, new_height))
    pixels = new_image.load()
    original_pixels = image.load()

    for y in range(new_height):
        for x in range(new_width):
            # Calculate nearest pixel position in the original image
            original_x = int(x * original_width / new_width)
            original_y = int(y * original_height / new_height)
            pixels[x, y] = original_pixels[original_x, original_y]

    return new_image

# Bilinear Interpolation
def bilinear_interpolation(image, new_width, new_height):
    original_width, original_height = image.size
    new_image = Image.new(image.mode, (new_width, new_height))
    pixels = new_image.load()
    original_pixels = image.load()

    for y in range(new_height):
        for x in range(new_width):
            # Calculate the position in the original image
            original_x = x * (original_width - 1) / (new_width - 1)
            original_y = y * (original_height - 1) / (new_height - 1)

            x0 = int(original_x)
            y0 = int(original_y)
            x1 = min(x0 + 1, original_width - 1)
            y1 = min(y0 + 1, original_height - 1)

            # Interpolation weights
            wx = original_x - x0
            wy = original_y - y0

            # Bilinear interpolation
            pixel = (
                (1 - wx) * (1 - wy) * np.array(original_pixels[x0, y0]) +
                wx * (1 - wy) * np.array(original_pixels[x1, y0]) +
                (1 - wx) * wy * np.array(original_pixels[x0, y1]) +
                wx * wy * np.array(original_pixels[x1, y1])
            )
            pixels[x, y] = tuple(pixel.astype(int))

    return new_image

# Streamlit UI
st.sidebar.success("Choose an algorithm")

st.title("Image Interpolation")

image = Image.open("images.jpg")
st.image(image, caption="Original Image", use_container_width=True)

    # User inputs for new dimensions
new_width = st.number_input("New width", min_value=1, value=image.size[0])
new_height = st.number_input("New height", min_value=1, value=image.size[1])

    # Interpolation method selection
method = st.selectbox("Choose Interpolation Method", ["Nearest Neighbor", "Bilinear"])

if st.button("Resize Image"):
    if method == "Nearest Neighbor":
        resized_image = nearest_neighbor_interpolation(image, new_width, new_height)
    else:
        resized_image = bilinear_interpolation(image, new_width, new_height)
    st.image(resized_image, caption=f"Resized Image ({method})", use_container_width=True)

