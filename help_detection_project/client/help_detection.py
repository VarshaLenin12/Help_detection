import tensorflow as tf
import cv2
import numpy as np
import time
import geocoder
import os
import requests
from uuid import uuid4
from datetime import datetime
import uuid

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Path to the SavedModel directory
model_path = r"C:\Users\varsh\Downloads\converted_savedmodel\model.savedmodel"

# Load the SavedModel
model = tf.saved_model.load(model_path)
infer = model.signatures['serving_default']

# Load the labels
labels_path = r"C:\Users\varsh\Downloads\converted_savedmodel\labels.txt"
with open(labels_path, "r") as file:
    class_names = {int(line.split()[0]): line.split()[1] for line in file.readlines()}

# Initialize the webcam
camera = cv2.VideoCapture(0)
time.sleep(2)

# Helper function to generate a unique filename
def generate_filename(file_extension):
    """Generate a unique filename with a timestamp and UUID."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex
    return f"{timestamp}_{unique_id}{file_extension}"

# Helper function to send data to the server
def send_to_servers(file_path, file_type, location):
    # URLs for different APIs
    api_urls = {
        "metadata_api": "https://androidapi220230605081325.azurewebsites.net/api/approval/AddSafetyAlerts",
        "local_server": "http://127.0.0.1:5000/upload"
    }
    
    # Prepare metadata
    current_time = get_current_time()
    filename = generate_filename(".jpg" if file_type == 'image' else ".mp4")
    location_str = f"{location[0]},{location[1]}"
    
    metadata = {
        "name": "Varsh",
        "location": location_str,
        "dateTime": current_time,
        "type": file_type,
        "fileName": filename
    }

    # For local server, include file in the request
    files = {
        'file': (filename, open(file_path, 'rb'), 'image/jpeg' if file_type == 'image' else 'video/mp4')
    }

    # Send data to the Android API (metadata only)
    try:
        response = requests.post(api_urls["metadata_api"], json=metadata)
        print(f"Sending metadata to {api_urls['metadata_api']}...")
        print("Metadata:", metadata)
        print("Response Status Code:", response.status_code)
        print("Response Content:", response.text)
        if response.status_code == 200 or response.status_code == 201:
            print(f"Metadata sent successfully to {api_urls['metadata_api']}!")
        else:
            print(f"Error on {api_urls['metadata_api']}:", response.status_code)
    except requests.RequestException as e:
        print(f"Exception while sending metadata to {api_urls['metadata_api']}: {e}")

    # Send data to the local server (metadata and file)
    try:
        response = requests.post(api_urls["local_server"], data=metadata, files=files)
        print(f"Sending file and metadata to {api_urls['local_server']}...")
        print("Metadata:", metadata)
        print("Response Status Code:", response.status_code)
        print("Response Content:", response.text)
        if response.status_code == 200 or response.status_code == 201:
            print(f"File and metadata sent successfully to {api_urls['local_server']}!")
        else:
            print(f"Error on {api_urls['local_server']}:", response.status_code)
    except requests.RequestException as e:
        print(f"Exception while sending data to {api_urls['local_server']}: {e}")



# Function to get current location
def get_current_location():
    g = geocoder.ip('me')
    return g.latlng if g.ok else (None, None)

# Function to get current time
def get_current_time():
    return datetime.now().strftime('%m/%d/%Y %H:%M:%S')

# Track the number of help signs detected and email sending state
help_sign_count = 0
email_sent = False

# Get the frame rate of the webcam
fps = camera.get(cv2.CAP_PROP_FPS) or 10  # Default to 10 if FPS is zero or not available

confidence_threshold = 0.8

while True:
    ret, image = camera.read()
    if not ret:
        print("Failed to capture image")
        break

    # Resize and preprocess the image
    resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1

    # Predict using the model
    prediction = infer(tf.convert_to_tensor(image_array, dtype=tf.float32))
    output_key = list(prediction.keys())[0]  # Get the first key
    prediction = prediction[output_key].numpy()
    index = np.argmax(prediction)
    class_name = class_names.get(index, "Unknown")
    confidence_score = prediction[0][index]

    # Get current location and time
    location = get_current_location()
    current_time = get_current_time()

    # Check if the "help" sign is detected
    if class_name == "help" and confidence_score >= confidence_threshold:
        help_sign_count += 1
        if help_sign_count == 1 and not email_sent:
            # Save the image for sending
            image_path = generate_filename(".jpg")
            # Overlay the location and time information on the image
            cv2.putText(image, f"Help sign detected! {current_time}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Location: Lat {location[0]}, Lon {location[1]}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imwrite(image_path, image)

            # Send the image and video to the servers
            send_to_servers(image_path, 'image', location)

            email_sent = True  # Set flag to prevent sending multiple data

            # Start capturing video and send follow-up video
            video_path = generate_filename(".mp4")

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # Or another codec that works for you
            fps = 20  # Adjust as needed
            frame_size = (int(camera.get(3)), int(camera.get(4)))  # Frame size

            video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
            
            start_time = time.time()

            # Record video for 30 seconds
            while time.time() - start_time < 30:
                ret, frame = camera.read()
                if ret:
                    video_writer.write(frame)
                else:
                    break

            video_writer.release()

            # Send video to the servers
            send_to_servers(video_path, 'video', location)

            # Clean up the video file
            os.remove(video_path)

            break  # Exit the loop after sending the video

        # Show the image with the text on it (Help sign detected)
        cv2.imshow("Webcam Image", image)

    else:
        help_sign_count = 0  # Reset count if the help sign is not detected

    # Show the live webcam feed
    cv2.imshow("Webcam Feed", image)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
