import cv2
from deepface import DeepFace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Força uso da CPU

img = cv2.imread("teste.jpeg")
result = DeepFace.analyze(
    img_path=img,
    actions=["age", "gender", "race"],
    detector_backend="opencv",
    enforce_detection=False,
)
print(result)
