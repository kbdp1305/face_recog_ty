import cv2
import tkinter as tk
from tkinter import messagebox, simpledialog
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils.databases import FaceEmbeddingDB
from utils.distance_counting import DistanceCounting
from src.face_recognition import FaceRecog

db_uri = "mongodb+srv://dharmaworkdev:dharma123@facerecognitiontrial.jxq2i.mongodb.net/?retryWrites=true&w=majority&appName=FaceRecognitionTrial"
db_name, collection_name = 'trial_acces_plany', 'mydatabase'

database = FaceEmbeddingDB(db_uri, db_name, collection_name)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, 
              thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True)
face_extractor = InceptionResnetV1(pretrained="vggface2").eval()
face_recognizer = FaceRecog(mtcnn, face_extractor)

def capture_face():
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture photo...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Capture Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return frame

def add_face():
    name = simpledialog.askstring("Input", "Enter name:")
    pos = simpledialog.askinteger("Input", "Enter position (1, 2, or 3):")
    if not name or pos not in [1, 2, 3]:
        messagebox.showerror("Error", "Invalid input!")
        return
    
    frame = capture_face()
    embedding, _ = face_recognizer.detect(frame)
    if embedding is None:
        messagebox.showerror("Error", "No face detected!")
        return
    
    database.add_data(name, pos, embedding)
    messagebox.showinfo("Success", "Face added successfully!")

def delete_face():
    pos = simpledialog.askstring("Input", "Enter Name to Delete :")
    database.delete_by_name(pos)
    messagebox.showinfo("Success", "Face data deleted!")

def update_face():
    pos = simpledialog.askinteger("Input", "Enter position to update (1, 2, or 3):")
    if pos not in [1, 2, 3]:
        messagebox.showerror("Error", "Invalid position!")
        return
    
    

    frame = capture_face()
    embedding, _ = face_recognizer.detect(frame)
    if embedding is None:
        messagebox.showerror("Error", "No face detected!")
        return
    
    database.update_embedding(pos, embedding)
    messagebox.showinfo("Success", "Face updated successfully!")

def recognize_face():
    pos = simpledialog.askinteger("Input", "Enter position to recognize (1, 2, or 3):")
    stored_embeddings = database.load_embeddings(pos)
    if not stored_embeddings:
        messagebox.showerror("Error", "No matching data found.")
        return
    
    frame = capture_face()
    new_embedding, _ = face_recognizer.detect(frame)
    if new_embedding is None:
        messagebox.showerror("Error", "No face detected!")
        return
    
    best_match = max(stored_embeddings, key=lambda data: DistanceCounting.cosine_similarity(data['tensor'], new_embedding), default=None)
    similarity = DistanceCounting.cosine_similarity(best_match['tensor'], new_embedding) if best_match else 0
    print(similarity)
    if similarity > 0.66:
        messagebox.showinfo("Access Granted", f"Welcome, {best_match['name']}!")
    else:
        messagebox.showerror("Access Denied", "Face does not match.")

def create_gui():
    root = tk.Tk()
    root.title("Face Recognition System")
    
    tk.Button(root, text="Add Face", command=add_face).pack(pady=10)
    tk.Button(root, text="Delete Face", command=delete_face).pack(pady=10)
    tk.Button(root, text="Update Face", command=update_face).pack(pady=10)
    tk.Button(root, text="Recognize Face", command=recognize_face).pack(pady=10)
    
    root.mainloop()

create_gui()
