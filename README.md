# Basic Face Recognition System

A simple, beginner-friendly face recognition application built with Python. The system allows users to register their face, train a model, and then perform real-time recognition using a webcam. All user data is stored in a local SQLite database.

---

## Features

-   **User Registration**: Capture face samples via webcam and store them.
-   **Database Integration**: Saves user names and IDs in a SQLite database.
-   **Model Training**: Trains a LBPH (Local Binary Patterns Histograms) face recognizer on the captured images.
-   **Real-Time Recognition**: Identifies registered users in a live webcam feed and displays their name.
-   **Simple GUI**: Easy-to-use graphical interface built with Tkinter.

---

## Tech Stack

-   **Language**: Python 3
-   **Computer Vision**: OpenCV (`opencv-contrib-python`)
-   **GUI**: Tkinter
-   **Database**: SQLite 3
-   **Image Processing**: Pillow (PIL) & NumPy

---
## Project Structure
- ├── dataset/              # Stores the captured face images for training.
- ├── trainer/              # Stores the trained model file (trainer.yml).
- ├── app.py                # The main Python script with the Tkinter GUI and recognition logic.
- ├── train_model.py        # The script used to train the face recognizer.
- ├── face_recognition.db   # The SQLite database file for storing user info.
- └── .gitignore            # Specifies files for Git to ignore.
