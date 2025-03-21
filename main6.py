import cv2
import tkinter as tk
from tkinter import messagebox
import time
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils.databases import FaceEmbeddingDB, InspectionDB
from utils.distance_counting import DistanceCounting
from src.face_recognition import FaceRecog
from PIL import Image, ImageTk
from datetime import datetime

# Database configurations
db_uri = "mongodb+srv://dharmaworkdev:dharma123@facerecognitiontrial.jxq2i.mongodb.net/?retryWrites=true&w=majority&appName=FaceRecognitionTrial"
db_name, collection_name = 'trial_acces_plany', 'mydatabase'
db_params = {
    "host": "localhost",
    "dbname": "postgres",
    "user": "postgres",
    "password": "123456",
    "port": 5432
}

# Initialize database and face recognition components
face_db = InspectionDB(db_params)
database = FaceEmbeddingDB(db_uri, db_name, collection_name)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, 
              thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True)
face_extractor = InceptionResnetV1(pretrained="vggface2").eval()
face_recognizer = FaceRecog(mtcnn, face_extractor)

def capture_face():
    cap = cv2.VideoCapture(0)
    print("Align your face inside the box.")
    cv2.namedWindow("Capture Face", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Capture Face", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    countdown = 3
    countdown_start = time.time()
    
    while countdown > 0:
        ret, frame = cap.read()
        if not ret:
            continue
        
        height, width, _ = frame.shape
        box_size = 450
        box_x, box_y = (width - box_size) // 2, (height - box_size) // 2
        
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_size, box_y + box_size), (0, 255, 0), 2)
        elapsed_time = time.time() - countdown_start
        
        if elapsed_time >= 1:
            countdown -= 1
            countdown_start = time.time()
        
        cv2.putText(frame, str(countdown), (width // 2 - 20, height // 2 - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow("Capture Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    return frame

def recognize_face():
    pos = 1
    stored_embeddings = database.load_embeddings(pos)
    if not stored_embeddings:
        messagebox.showerror("Error", "No matching data found.")
        return None, None
    
    frame = capture_face()
    new_embedding, _ = face_recognizer.detect(frame)
    if new_embedding is None:
        messagebox.showerror("Error", "No face detected!")
        return None, None
    
    best_match = max(stored_embeddings, key=lambda data: DistanceCounting.cosine_similarity(data['tensor'], new_embedding), default=None)
    similarity = DistanceCounting.cosine_similarity(best_match['tensor'], new_embedding) if best_match else 0
    best_name = best_match["name"] if best_match else None
    
    full_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    full_screen[:] = (0, 255, 0) if similarity > 0.66 else (0, 0, 255)
    
    verified_img_path = r"C:\Magang\Toyota\face_recognition\dataset\biruu.png" if similarity > 0.66 else r"C:\Magang\Toyota\face_recognition\dataset\xmark.png"
    verified_img = cv2.imread(verified_img_path, cv2.IMREAD_UNCHANGED)
    
    if verified_img is not None and verified_img.shape[2] == 4:
        bgr, alpha = verified_img[:, :, :3], verified_img[:, :, 3] / 255.0
        bgr, alpha = cv2.resize(bgr, (300, 300)), cv2.resize(alpha, (300, 300))
        x_offset, y_offset = (1920 - 300) // 2, (1080 - 300) // 2
        roi = full_screen[y_offset:y_offset+300, x_offset:x_offset+300]
        full_screen[y_offset:y_offset+300, x_offset:x_offset+300] = ((1 - alpha[:, :, None]) * roi + alpha[:, :, None] * bgr).astype(np.uint8)
    
    text = "Face Verified" if similarity > 0.66 else "Face Not Verified"
    cv2.putText(full_screen, text, (800, 900), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5, cv2.LINE_AA)
    
    cv2.imshow("Recognition Result", full_screen)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S") if similarity > 0.66 else None, best_name

def on_recognize(label):
    now, best_name = recognize_face()
    if now:
        label.config(text=f"{now}")
        face_db.add_stamp(best_name, now)

def create_gui():
    root = tk.Tk()
    root.title("Face Recognition System")
    root.state("zoomed")
    
    bg_photo = ImageTk.PhotoImage(Image.open(r"C:\Magang\Toyota\face_recognition\dataset\qis.jpg").resize((root.winfo_screenwidth(), root.winfo_screenheight())))
    tk.Label(root, image=bg_photo).place(relwidth=1, relheight=1)
    
    label = tk.Label(root, text="", font=("Arial", 16, "bold"), fg="black", bg='#FAFAFA')
    label.place(relx=0.92, rely=0.11, anchor=tk.CENTER)
    
    button_photo = ImageTk.PhotoImage(Image.open(r"C:\Magang\Toyota\face_recognition\dataset\stamp.png").resize((248, 42), Image.Resampling.LANCZOS))
    tk.Button(root, image=button_photo, borderwidth=0, command=lambda: on_recognize(label)).place(relx=0.35, rely=0.982, anchor=tk.CENTER)
    
    root.mainloop()

create_gui()