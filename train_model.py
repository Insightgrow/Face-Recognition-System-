import cv2
import numpy as np
from PIL import Image # Pillow library
import os

def train_model():
    """
    Trains the LBPH face recognizer using the images in the 'dataset' folder.
    It reads each image, associates it with a user ID, and then trains the model.
    The result is saved as 'trainer.yml'.
    """
    # Path for the folder containing face images
    path = 'dataset'
    
    # This is the face recognizer we will use
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # This is the face detector
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def get_images_and_labels(path):
        """
        This function gets all the image paths from the dataset folder
        and extracts the user ID from the filename.
        """
        # Get all file paths in the dataset folder
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            # Open the image using Pillow and convert it to grayscale
            pil_image = Image.open(image_path).convert('L') # 'L' stands for Luminance (grayscale)
            img_numpy = np.array(pil_image, 'uint8') # Convert the image to a NumPy array

            # Extract the user ID from the filename
            # The filename format is: user.ID.sampleNumber.jpg
            try:
                id = int(os.path.split(image_path)[-1].split(".")[1])
            except (IndexError, ValueError):
                print(f"Skipping file with incorrect format: {image_path}")
                continue
            
            # We don't need to detect faces here again, as we saved only face regions.
            # For simplicity, we assume they are.
            face_samples.append(img_numpy)
            ids.append(id)

        return face_samples, ids

    print("\n[INFO] Training faces. This might take a moment...")
    faces, ids = get_images_and_labels(path)
    
    if len(ids) == 0:
        print("\n[INFO] No faces found to train. Please register faces first using the main app.")
        return

    # Train the recognizer with our face images and their corresponding IDs
    recognizer.train(faces, np.array(ids))

    # Create the 'trainer' directory if it doesn't exist
    if not os.path.exists('trainer'):
        os.makedirs('trainer')
        
    # Save the trained model to a file
    recognizer.write('trainer/trainer.yml')

    # Print the number of unique faces trained
    print(f"\n[INFO] {len(np.unique(ids))} faces trained successfully. The model is saved as trainer/trainer.yml")

if __name__ == "__main__":
    train_model()
