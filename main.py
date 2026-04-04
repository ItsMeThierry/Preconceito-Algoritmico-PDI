import cv2
import os

detector = cv2.FaceDetectorYN.create(
    model='face_detection_yunet_2023mar_int8.onnx', 
    config="", 
    input_size=(320, 320), # Tamanho da imagem de rosto
    score_threshold=0.85, # Confiança mínima, o ideal seria entre (0.6-0.9)
    nms_threshold=0.3, # (0.3-0.5)
    top_k=5000, # Número máximo de rostos detectado
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU # Executa o modelo na CPU
)

def extract_faces(img):

    # Definindo o tamanho da imagem no detector
    height, width = img.shape[:2]
    detector.setInputSize((width, height))

    # Detecta faces
    print("Detectando faces...")
    faces = detector.detect(img)

    collection = list()

    # Extrai as faces
    for face in faces[1]:
        x, y, w ,h = map(int, face[:4])
        confianca = face[14]

        print(f"Extraindo face x = {x}, y = {y}, w = {w}, h = {h}, confianca = {confianca}")
        
        face_roi = img[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

        collection.append(face_rgb)

    return collection
        

def main():
    
    extensoes = {'.png', '.jpg', '.jpeg'}
    path_name = None
    extensao = None

    # Selecionar imagem
    while True:
        path_name = input("Selecione uma imagem na pasta imagens (sem extensão): ")

        for ext in extensoes:
            if os.path.exists('imagens/' + path_name + ext):
                extensao = ext

        if extensao:
            break

        print(f"O arquivo {path_name} não existe")
        continue

    try:
        img = cv2.imread('imagens/' + path_name + extensao)

        # Extrai todas as faces da imagem
        faces_img = extract_faces(img)

        if not os.path.exists("output"):
            os.makedirs("output")

        # Salva todas as faces em imagens separadas
        i = 0
        for face in faces_img:
            cv2.imwrite(f"output/{path_name}_face_{i}.jpg", face)
            i += 1
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main() 