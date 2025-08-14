import tkinter as tk
from tkinter import messagebox
import cv2
import os
import sqlite3

class FaceRecognitionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("500x320") # Adjusted size slightly
        self.window.configure(bg='#eaf2f8') # A light blue background

        # --- Database Setup ---
        # Connect to SQLite database (or create it if it doesn't exist)
        self.conn = sqlite3.connect('face_recognition.db')
        self.cursor = self.conn.cursor()
        # Create a table to store user ID and name if it's not already there
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
        ''')
        self.conn.commit()

        # --- UI Elements ---
        # Main title label
        self.label = tk.Label(window, text="Face Recognition System", font=("Arial", 20, "bold"), bg='#eaf2f8', fg='#34495e')
        self.label.pack(pady=20)

        # Label and entry box for the user's name
        self.name_label = tk.Label(window, text="Enter Your Name:", font=("Arial", 12), bg='#eaf2f8')
        self.name_label.pack()
        self.name_entry = tk.Entry(window, font=("Arial", 12), width=30, bd=2, relief=tk.GROOVE)
        self.name_entry.pack(pady=5)

        # --- Buttons ---
        # Button to start registering a new face
        self.register_button = tk.Button(window, text="Register New Face", command=self.register_face, font=("Arial", 12, "bold"), bg='#2ecc71', fg='white', relief=tk.RAISED, bd=3)
        self.register_button.pack(pady=10)
        
        # Button to start the training process
        self.train_button = tk.Button(window, text="Train Model", command=self.train_model_gui, font=("Arial", 12, "bold"), bg='#f39c12', fg='white', relief=tk.RAISED, bd=3)
        self.train_button.pack(pady=5)

        # Button to start recognizing faces
        self.recognize_button = tk.Button(window, text="Recognize Faces", command=self.recognize_faces, font=("Arial", 12, "bold"), bg='#3498db', fg='white', relief=tk.RAISED, bd=3)
        self.recognize_button.pack(pady=5)

        # Create the 'dataset' directory if it doesn't exist. This is where we'll save face images.
        if not os.path.exists('dataset'):
            os.makedirs('dataset')

        self.window.mainloop()

    def register_face(self):
        """
        Captures 50 face images from the webcam for a new user.
        """
        name = self.name_entry.get()
        if not name:
            messagebox.showerror("Error", "Please enter a name.")
            return

        try:
            # Insert user into the database. The 'id' is generated automatically.
            self.cursor.execute("INSERT INTO users (name) VALUES (?)", (name,))
            self.conn.commit()
            user_id = self.cursor.lastrowid # Get the ID of the new user
            messagebox.showinfo("Info", f"Registering {name} with ID: {user_id}.\nPlease look at the camera. The process will start in 3 seconds.")
            self.window.update()
            cv2.waitKey(3000) # Wait for 3 seconds
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", f"The name '{name}' already exists. Please use a different name.")
            return

        # --- Face Capture Logic ---
        # Load the pre-trained model for face detection from OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        video_capture = cv2.VideoCapture(0) # Start the webcam
        
        count = 0
        while count < 50: # We will capture 50 images
            ret, frame = video_capture.read() # Read a frame from the webcam
            if not ret:
                messagebox.showerror("Error", "Could not access webcam.")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale (better for detection)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw a rectangle around the face
                count += 1
                # Save the captured face image in the 'dataset' folder
                # The filename format is important: user.ID.sampleNumber.jpg
                cv2.imwrite(f"dataset/user.{user_id}.{count}.jpg", gray[y:y+h, x:x+w])
                # Display the current frame in a window
                cv2.imshow('Capturing Faces...', frame)

            # Press 'q' to stop the process early
            if cv2.waitKey(100) & 0xFF == ord('q'): # Wait 100ms between captures
                break
        
        video_capture.release() # Release the webcam
        cv2.destroyAllWindows() # Close the image window
        messagebox.showinfo("Success", f"{count} images captured for {name}.")
        self.name_entry.delete(0, 'end') # Clear the name entry box

    def train_model_gui(self):
        """
        Informs the user to run the separate training script.
        """
        messagebox.showinfo("Train Model", "To train the model, please close this app and run the 'train_model.py' script from your VS Code terminal.")

    def recognize_faces(self):
        """
        Starts the webcam to recognize faces based on the trained model.
        """
        # Check if the trained model file exists
        if not os.path.exists('trainer/trainer.yml'):
            messagebox.showerror("Error", "Model not trained yet! Please register at least one face and then run the 'train_model.py' script.")
            return
            
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml') # Load the trained model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        video_capture = cv2.VideoCapture(0)
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for(x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                # The recognizer predicts the ID and confidence level for the face
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                
                # Confidence is a measure of distance, so lower is better. 0 is a perfect match.
                if confidence < 75: # We can set a threshold
                    # Fetch the user's name from the database using the predicted ID
                    self.cursor.execute("SELECT name FROM users WHERE id = ?", (id,))
                    result = self.cursor.fetchone()
                    name = result[0] if result else "Unknown"
                    confidence_text = f"  {round(100 - confidence)}%"
                else:
                    name = "Unknown"
                    confidence_text = ""

                # Display the name and confidence on the screen
                cv2.putText(frame, str(name), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, str(confidence_text), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)  
            
            cv2.imshow('Face Recognition (Press q to exit)',frame) 
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        video_capture.release()
        cv2.destroyAllWindows()

# This part runs the app
if __name__ == "__main__":
    # Create the main window and start the application
    FaceRecognitionApp(tk.Tk(), "Face Recognition System")
