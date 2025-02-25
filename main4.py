import cv2
import tkinter as tk
from tkinter import messagebox, simpledialog
import time
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils.databases import FaceEmbeddingDB
from utils.distance_counting import DistanceCounting
from src.face_recognition import FaceRecog
import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk
from datetime import datetime



db_uri = "mongodb+srv://dharmaworkdev:dharma123@facerecognitiontrial.jxq2i.mongodb.net/?retryWrites=true&w=majority&appName=FaceRecognitionTrial"
db_name, collection_name = 'trial_acces_plany', 'mydatabase'
database = FaceEmbeddingDB(db_uri, db_name, collection_name)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, 
              thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True)
face_extractor = InceptionResnetV1(pretrained="vggface2").eval()
face_recognizer = FaceRecog(mtcnn, face_extractor)

import cv2
import time

def capture_face():
    cap = cv2.VideoCapture(0)
    print("Align your face inside the box. Photo will be captured in 3 seconds...")
    cv2.namedWindow("Capture Face", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Capture Face", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Get frame dimensions
        height, width, _ = frame.shape
        # Define box size and position
        box_size = 450
        box_x = (width - box_size) // 2
        box_y = (height - box_size) // 2
        
        # Draw rectangle (face guide)
        color = (0, 255, 0)  # Green
        thickness = 2
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_size, box_y + box_size), color, thickness)
        
        # Show frame
        cv2.imshow("Capture Face", frame)
        
        # Capture conditions
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
        if time.time() - start_time >= 3:
            print("Photo captured!")
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame


def recognize_face():
    pos = 1
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
    
    full_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    height, width, _ = full_screen.shape

    if similarity > 0.66:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current date & time

        full_screen[:] = (0, 255, 0)  # Green background
        # cv2.putText(full_screen, "", (width // 2 - 300, height // 2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 15, (255, 255, 255), 30)
        
        verified_img = cv2.imread(r"C:\Magang\Toyota\face_recognition\dataset\biruu.png", cv2.IMREAD_UNCHANGED)
        if verified_img is not None:
            if verified_img.shape[2] == 4:  # Jika ada channel alpha (transparan)
                bgr = verified_img[:, :, :3]  # Ambil channel warna (BGR)
                alpha = verified_img[:, :, 3]  # Ambil channel alpha

                # Resize gambar yang akan ditempel
                bgr = cv2.resize(bgr, (300, 300))
                alpha = cv2.resize(alpha, (300, 300))

                vh, vw, _ = bgr.shape
                x_offset = (width - vw) // 2
                y_offset = (height - vh) // 2

                # Buat region of interest (ROI) di latar belakang
                roi = full_screen[y_offset:y_offset+vh, x_offset:x_offset+vw]

                # Gunakan alpha mask untuk blending
                alpha = alpha[:, :, np.newaxis] / 255.0  # Normalisasi alpha ke [0, 1]
                roi = (1 - alpha) * roi + alpha * bgr  # Blending gambar
                full_screen[y_offset:y_offset+vh, x_offset:x_offset+vw] = roi.astype(np.uint8)
                text = "Face Verified"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2
                font_thickness = 5
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                
                text_x = (width - text_size[0]) // 2
                text_y = y_offset + vh + 50  # Posisi di bawah gambar biru
                

                cv2.putText(full_screen, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                return now
                

    else:
        full_screen[:] = (0, 0, 255)  # Green background
        not_verified_img = cv2.imread(r"C:\Magang\Toyota\face_recognition\dataset\xmark.png", cv2.IMREAD_UNCHANGED)
        if not_verified_img is not None:
            if not_verified_img.shape[2] == 4:  # Jika ada channel alpha (transparan)
                bgr = not_verified_img[:, :, :3]  # Ambil channel warna (BGR)
                alpha = not_verified_img[:, :, 3]  # Ambil channel alpha

                # Resize gambar yang akan ditempel
                bgr = cv2.resize(bgr, (300, 300))
                alpha = cv2.resize(alpha, (300, 300))

                vh, vw, _ = bgr.shape
                x_offset = (width - vw) // 2
                y_offset = (height - vh) // 2

                # Buat region of interest (ROI) di latar belakang
                roi = full_screen[y_offset:y_offset+vh, x_offset:x_offset+vw]

                # Gunakan alpha mask untuk blending
                alpha = alpha[:, :, np.newaxis] / 255.0  # Normalisasi alpha ke [0, 1]
                roi = (1 - alpha) * roi + alpha * bgr  # Blending gambar
                full_screen[y_offset:y_offset+vh, x_offset:x_offset+vw] = roi.astype(np.uint8)
                text = "Face Not Verified"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2
                font_thickness = 5
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                
                text_x = (width - text_size[0]) // 2
                text_y = y_offset + vh + 50  # Posisi di bawah gambar biru

                cv2.putText(full_screen, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                # return 0
        # full_screen[:] = (0, 0, 255)  # Red background
        # cv2.putText(full_screen, "X", (width // 2 - 300, height // 2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 15, (255, 255, 255), 30)

    cv2.namedWindow("Recognition Result", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Recognition Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Recognition Result", full_screen)
    cv2.waitKey(3000)  # Show for 3 seconds
    cv2.destroyAllWindows()


# def create_gui():
#     root = tk.Tk()
#     root.title("Face Recognition System")
#     root.state("zoomed")  # Make GUI full screen
    
#     tk.Button(root, text="Recognize Face", command=recognize_face, font=("Arial", 24), height=3, width=20).pack(pady=20)
    
#     root.mainloop()

# create_gui()

def on_recognize(label):
    """Calls recognize_face() and updates the label if verified."""
    result = recognize_face()
    if result:
        label.config(text=f"Face Recognized!\n{result}")


def create_gui():
    root = tk.Tk()
    root.title("Face Recognition System")
    root.state("zoomed")  # Make GUI full screen
    
    # Load background image
    bg_image = Image.open(r"C:\Magang\Toyota\face_recognition\dataset\qis.jpg")  # Ganti dengan path gambar yang sesuai
    bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()))
    bg_photo = ImageTk.PhotoImage(bg_image)
    
    # Set background
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)
    # Load the button image
    button_image = Image.open(r"C:\Magang\Toyota\face_recognition\dataset\stamp.png")  # Replace with your image path
    button_image = button_image.resize((248, 42), Image.Resampling.LANCZOS)  # Resize if needed
    button_photo = ImageTk.PhotoImage(button_image)
    # Create Canvas for rectangle and date input
    # canvas = tk.Canvas(root, width=150, height=45, bg="white", highlightthickness=0)
        # Create Canvas (no background for transparency effect)
    # canvas = tk.Canvas(root, width=150, height=45, highlightthickness=0,bg=root["bg"])
    # # canvas.place(relx=0.75, rely=0.9, anchor=tk.CENTER)  # Adjust as needed
    # canvas.place(relx=0.9, rely=0.11, anchor=tk.CENTER)  # Adjust position as needed

    # Draw a rectangle
    # canvas.create_rectangle(5, 5, 295, 75, outline="black", width=2)
    # canvas.create_text(150, 20, text="Face Recognized Date:", font=("Arial", 14, "bold"))
    label = tk.Label(root, text="", font=("Arial", 16, "bold"), fg="black", bg='#FAFAFA')
    label.place(relx=0.92, rely=0.11, anchor=tk.CENTER)
    # global date_entry
    # date_entry = tk.Entry(root, font=("Arial", 14), justify="center", bd=0, highlightthickness=0)
    # canvas.create_window(75, 22, window=date_entry)  # Centered inside the Canvas
    # Create an image button
    btn = tk.Button(root, image=button_photo, borderwidth=0, command=lambda: on_recognize(label))

    # btn = tk.Button(root, image=button_photo, command=recognize_face, borderwidth=0)  # Remove border
    btn.place(relx=0.35, rely=0.982, anchor=tk.CENTER)  # Adjust position as needed
    # Button on top of background
    # btn = tk.Button(root, text="Recognize Face", command=recognize_face, font=("Arial", 24), height=1, width=10)
    # btn.place(relx=0.1, rely=0.1, anchor=tk.CENTER)
    
    root.mainloop()

create_gui()
