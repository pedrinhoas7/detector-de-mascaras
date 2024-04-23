import cv2 as cv
import functions

# Configurações da Webcam
CAMERA_INDEX = 0
FULL_HD_WIDTH = 1920
FULL_HD_HEIGHT = 1080

# Configurações do Classificador
CASCADE_PATH = f"{cv.data.haarcascades}/haarcascade_frontalface_alt2.xml"
FACE_MIN_SIZE = (200, 200) # Tamanho mínimo da face para reconhecimento
RESIZED_FACE_DIMS = (160, 160) # Dimensões para redimensionar a face antes de aplicar PCA

# Configurações de detecção e classificação
PCA_FEATURES = 30  # Número de componentes principais
KNN_NEIGHBORS = 3   # Número de vizinhos mais próximos em KNN

# Carregar dados e modelos
dataframe = functions.load_dataframe()
X_train, X_test, y_train, y_test = functions.train_test(dataframe)
pca = functions.pca_model(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
knn = functions.knn(X_train, y_train)

# Inicializando a câmera
cam = cv.VideoCapture(CAMERA_INDEX)
cam.set(cv.CAP_PROP_FRAME_WIDTH, FULL_HD_WIDTH)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, FULL_HD_HEIGHT)

# Carregando classificador de Haar
classifier = cv.CascadeClassifier(CASCADE_PATH)

# Rótulos
labels = {0: "Sem mascara", 1: "Com mascara"}

# Processamento de frames
while True:
    status, frame = cam.read()
    if not status:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, minSize=FACE_MIN_SIZE)

    for x, y, w, h in faces:
        gray_face = gray[y:y+h, x:x+w]
        gray_face = cv.resize(gray_face, RESIZED_FACE_DIMS)
        vector = pca.transform([gray_face.flatten()])
        pred = knn.predict(vector)[0]
        classification = labels[pred]

        color = (0, 255, 0) if pred == 1 else (0, 0, 255)
        cv.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv.putText(frame, classification, (x, y + h + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv.putText(frame, f"{len(faces)} rostos identificados", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv.imshow("Cam", frame)

    if cv.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
