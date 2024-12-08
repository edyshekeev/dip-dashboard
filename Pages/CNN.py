import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import (
    ResNet50,
    DenseNet121,
    InceptionV3,
)
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np
from PIL import Image
import os  # Importing os module

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to load the selected model
def load_model(model_name, input_shape=(224, 224, 3)):
    if model_name == "AlexNet":
        model = Sequential([
            Conv2D(96, kernel_size=11, strides=4, activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=3, strides=2),
            Conv2D(256, kernel_size=5, activation="relu"),
            MaxPooling2D(pool_size=3, strides=2),
            Conv2D(384, kernel_size=3, activation="relu"),
            Conv2D(384, kernel_size=3, activation="relu"),
            Conv2D(256, kernel_size=3, activation="relu"),
            MaxPooling2D(pool_size=3, strides=2),
            Flatten(),
            Dense(4096, activation="relu"),
            Dropout(0.5),
            Dense(4096, activation="relu"),
            Dropout(0.5),
            Dense(2, activation="softmax"),
        ])
    elif model_name == "LeNet":
        model = Sequential([
            Conv2D(6, kernel_size=5, activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=2),
            Conv2D(16, kernel_size=5, activation="relu"),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(120, activation="relu"),
            Dense(84, activation="relu"),
            Dense(2, activation="softmax"),
        ])
    elif model_name == "ResNet":
        model = ResNet50(weights=None, input_shape=input_shape, classes=2)
    elif model_name == "DenseNet":
        model = DenseNet121(weights=None, input_shape=input_shape, classes=2)
    elif model_name == "GoogleNet":
        model = InceptionV3(weights=None, input_shape=input_shape, classes=2)
    else:
        st.error("Unknown model selected!")
        return None
    return model

# Streamlit UI
st.title("Cat vs Dog Classifier")

st.sidebar.title("Model Selection")
model_name = st.sidebar.selectbox("Choose a CNN Model", ["AlexNet", "LeNet", "ResNet", "DenseNet", "GoogleNet"])

st.sidebar.title("Training and Testing")
train_data_dir = st.sidebar.text_input("Training Set Directory", "training_set")
test_data_dir = st.sidebar.text_input("Testing Set Directory", "test_set")

# Choose the predefined image
image_options = [
    "dataset/single_prediction/cat_or_dog_1.jpg",
    "dataset/single_prediction/cat_or_dog_2.jpg",
]
selected_image = st.selectbox("Select the predefined image to classify:", image_options)

if not os.path.exists(selected_image):
    st.error(f"Image file '{selected_image}' not found!")
else:
    st.image(selected_image, caption=f"Selected Image: {selected_image}", use_column_width=True)

if st.button("Train and Test Model"):
    try:
        # Prepare data
        datagen = ImageDataGenerator(rescale=1.0 / 255.0)
        train_generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
        )
        test_generator = datagen.flow_from_directory(
            test_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
        )

        # Load selected model
        model = load_model(model_name, input_shape=(224, 224, 3))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # Train model
        st.text("Training the model...")
        history = model.fit(train_generator, epochs=5, validation_data=test_generator)

        # Test model
        st.text("Testing the model...")
        test_loss, test_accuracy = model.evaluate(test_generator)
        st.success(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        # Prediction on selected image
        st.text("Predicting the selected image...")
        preprocessed_image = preprocess_image(selected_image, target_size=(224, 224))
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction, axis=1)
        classes = ["Cat", "Dog"]
        st.write(f"The selected image is classified as: **{classes[predicted_class[0]]}**")
    except Exception as e:
        st.error(f"An error occurred: {e}")
