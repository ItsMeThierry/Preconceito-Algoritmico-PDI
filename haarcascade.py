import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

def extract_faces(img):

    # Convertendo a imagem em cinza
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecta faces
    print("Detectando faces...")
    faces = face_cascade.detectMultiScale(img, 1.1, 5)

    collection = list()

    # Extrai as faces
    for (x, y, w, h) in faces:
        print(f"Extraindo face x = {x}, y = {y}, w = {w}, h = {h}")
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