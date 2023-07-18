import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import tensorflow as tf

# Load the trained eye detection model
model = tf.keras.models.load_model('mouth_detection_model.h5')

# Create a Tkinter window
window = tk.Tk()
window.title("Mouth Detection")
window.geometry("400x300")

# Create a label to display the predicted mouth state
label_result = tk.Label(window, text="", font=("Helvetica", 16))
label_result.pack(pady=20)

# Function to perform mouth detection on the selected image
def detect_mouth_state():
    # Open a file dialog to select an image
    image_path = filedialog.askopenfilename(title="Select Image", filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"),))

    # Read and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (24, 24))
    normalized_image = resized_image / 255.0
    input_data = np.reshape(normalized_image, (1, 24, 24, 1))

    # Perform mouth state prediction
    prediction = model.predict(input_data)
    mouth_state = "yawn" if prediction[0][1] > 0.5 else "no yawn"

    # Update the label with the predicted mouth state
    label_result.configure(text=f"Mouth state: {mouth_state}")

# Button to trigger the mouth detection
button_detect = tk.Button(window, text="Detect Mouth State", command=detect_mouth_state)
button_detect.pack(pady=10)

# Run the Tkinter main loop
window.mainloop()
