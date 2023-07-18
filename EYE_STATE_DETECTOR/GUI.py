import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import tensorflow as tf

# Load the trained eye detection model
model = tf.keras.models.load_model('eye_detection_model.h5')

# Create a Tkinter window
window = tk.Tk()
window.title("Eye Detection")
window.geometry("400x300")

# Create a label to display the predicted eye state
label_result = tk.Label(window, text="", font=("Helvetica", 16))
label_result.pack(pady=20)

# Function to perform eye detection on the selected image
def detect_eye_state():
    # Open a file dialog to select an image
    image_path = filedialog.askopenfilename(title="Select Image", filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"),))

    # Read and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (24, 24))
    normalized_image = resized_image / 255.0
    input_data = np.reshape(normalized_image, (1, 24, 24, 1))

    # Perform eye state prediction
    prediction = model.predict(input_data)
    eye_state = "Open" if prediction[0][1] > 0.5 else "Closed"

    # Update the label with the predicted eye state
    label_result.configure(text=f"Eye state: {eye_state}")

# Button to trigger the eye detection
button_detect = tk.Button(window, text="Detect Eye State", command=detect_eye_state)
button_detect.pack(pady=10)

# Run the Tkinter main loop
window.mainloop()
