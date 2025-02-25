from facenet_pytorch import MTCNN, InceptionResnetV1
from tensorflow.keras.preprocessing.image import load_img
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class FaceRecog:
   def __init__(self,mtcnn,facenet) :
       self.mtcnn=mtcnn
       self.facenet=facenet
   def preprocess_image(self,img):
    """Detect face and return embedding from an image file or camera frame."""
    
    if isinstance(img, str):  # If input is a file path
        img = load_img(img)  # Load image using Keras load_img()
    elif isinstance(img, np.ndarray):  # If input is a NumPy array (from OpenCV)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB
        img = Image.fromarray(img)  # Convert to PIL image
    else:
        raise TypeError("Input should be a file path or a NumPy array.")
    # Force image to RGB format
    img = img.convert("RGB")
    return img

   def detect(self,img) :
        face_tensor, _ = self.mtcnn(img, return_prob=False)
        # try:
        #     self.visualize_output(img, _)
        # except Exception as e:
        #     print(f"Error occurred: {e}")


        if face_tensor is None:
            print("No face detected.")
            return None
        
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
        embedding = self.facenet(face_tensor)  # Extract embedding
        return embedding,_

   def visualize_output(self,img1,img2) : 
        # Show cropped faces
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(img1)
            ax[0].set_title("Cropped Face 1")
            ax[0].axis("off")

            ax[1].imshow(img2)
            ax[1].set_title("Cropped Face 2")
            ax[1].axis("off")
            plt.show()
    

    
       
        