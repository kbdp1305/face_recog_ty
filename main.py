import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils.databases import FaceEmbeddingDB
from utils.distance_counting import DistanceCounting
from src.face_recognition import FaceRecog

def face_recognition(pos):
    # Database Configuration
    db_uri = "mongodb+srv://dharmaworkdev:dharma123@facerecognitiontrial.jxq2i.mongodb.net/?retryWrites=true&w=majority&appName=FaceRecognitionTrial"
    db_name, collection_name = 'trial_acces_plany', 'mydatabase'
    
    # Initialize Face Detection & Recognition Modelsc
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, 
                  thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True)
    face_extractor = InceptionResnetV1(pretrained="vggface2").eval()
    face_recognizer = FaceRecog(mtcnn, face_extractor)
    
    # Load Face Embeddings from Database
    database = FaceEmbeddingDB(db_uri, db_name, collection_name)
    stored_embeddings = database.load_embeddings(pos)
    if not stored_embeddings:
        print("No matching data found.")
        return
    
    # Capture Image from Webcam
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture photo...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Extract Face Embedding
    new_embedding, _ = face_recognizer.detect(frame)
    if new_embedding is None:
        print("No face detected. Try again.")
        return
    
    # Compare Embeddings
    best_match = max(stored_embeddings, key=lambda data: DistanceCounting.cosine_similarity(data['tensor'], new_embedding), default=None)
    similarity = DistanceCounting.cosine_similarity(best_match['tensor'], new_embedding) if best_match else 0
    
    # Authentication Result
    if similarity > 0.6:
        print(f"✅ Access Granted! Welcome, {best_match['name']}.")
    else:
        print("❌ Access Denied. Face does not match.")
    
    database.close()

# Run Face Recognition
face_recognition(3)