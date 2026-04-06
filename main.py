import cv2
import os
import json
from deepface import DeepFace

def analyze_faces(img):
    try:
        # Detecta as faces e extrai uma analise delas
        faces_data = DeepFace.analyze(img_path=img, actions=['age','gender','race'],detector_backend='mtcnn',enforce_detection=False)
        print(f"Detectado {len(faces_data)} faces")

        # Processa os dados do resultado em (face_img, data)
        collection = list()
        print("Processando os dados das faces...")

        for data in faces_data:
            x = data['region']['x']
            y = data['region']['y']
            w = data['region']['w']
            h = data['region']['h']

            # Região detectada da face
            face_region = img[y:y+h, x:x+w]

            # Dados importantes do resultado
            face_data = {
                'idade': data['age'],
                'confianca_face': data['face_confidence'],
                'prob_genero': {'homem': float(data['gender']['Man']), 'mulher': float(data['gender']['Woman'])},
                'prob_etnia': {
                    'asiatico': float(data['race']['asian']),
                    'indiano': float(data['race']['indian']),
                    'negro': float(data['race']['black']),
                    'branco': float(data['race']['white']),
                    'oriente_medio': float(data['race']['middle eastern']),
                    'latino': float(data['race']['latino hispanic'])
                }
            }

            collection.append((face_region, face_data))

        return collection

    except Exception as e:
        raise Exception(e)

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

        # Analise as faces das imagens e coleta dados sobre elas
        result = analyze_faces(img)

        if not os.path.exists("output"):
            os.makedirs("output")

        # Para cada face salva a imagem e os dados, separadamente
        i = 0
        print("Salvando as imagens e os dados em output...")
        for (face_img, data) in result:
            # Salva uma imagem da face
            cv2.imwrite(f"output/{path_name}_face_{i}.jpg", face_img)

            # Salva os dados da face
            with open(f"output/{path_name}_face_{i}.json", 'w') as f:
                json.dump(data, f)
        
            i += 1

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main() 