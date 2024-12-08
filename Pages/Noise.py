import streamlit as st
import numpy as np
from PIL import Image

# Function to add Gaussian Noise
def add_gaussian_noise(image, mean=0, var=0.01):
    image_np = np.array(image).astype(float) / 255.0  # Convert to [0, 1] range
    sigma = var**0.5
    gaussian_noise = np.random.normal(mean, sigma, image_np.shape)
    noisy_image = image_np + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 1)  # Ensure values remain within [0, 1] range
    noisy_image = (noisy_image * 255).astype(np.uint8)  # Convert back to [0, 255]
    return Image.fromarray(noisy_image)

# Function to add Salt and Pepper Noise
def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    image_np = np.array(image)
    noisy_image = np.copy(image_np)

    # Salt noise
    num_salt = np.ceil(salt_prob * image_np.size).astype(int)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image_np.shape]
    noisy_image[coords[0], coords[1], :] = 255

    # Pepper noise
    num_pepper = np.ceil(pepper_prob * image_np.size).astype(int)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image_np.shape]
    noisy_image[coords[0], coords[1], :] = 0

    return Image.fromarray(noisy_image)

# Streamlit UI
st.sidebar.success("Choose a noise type")

st.title("Image Noise Addition")

image = Image.open("images.jpg")
st.image(image, caption="Original Image", use_container_width=True)

    # Noise selection
noise_type = st.selectbox("Choose Noise Type", ["Gaussian Noise", "Salt-Pepper Noise"])

if noise_type == "Gaussian Noise":
    mean = st.slider("Mean", 0.0, 1.0, 0.0)
    var = st.slider("Variance", 0.0, 1.0, 0.01)
    if st.button("Apply Gaussian Noise"):
        noisy_image = add_gaussian_noise(image, mean, var)
        st.image(noisy_image, caption="Image with Gaussian Noise", use_container_width=True)
    
elif noise_type == "Salt-Pepper Noise":
    salt_prob = st.slider("Salt Probability", 0.0, 0.1, 0.01)
    pepper_prob = st.slider("Pepper Probability", 0.0, 0.1, 0.01)
    if st.button("Apply Salt-Pepper Noise"):
        noisy_image = add_salt_pepper_noise(image, salt_prob, pepper_prob)
        st.image(noisy_image, caption="Image with Salt-Pepper Noise", use_container_width=True)