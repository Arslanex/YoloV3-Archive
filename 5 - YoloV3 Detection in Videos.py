import cv2 as cv
import numpy as np

# Kamera ayarlamaları
cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Model Parametreleri
inputSize = 0                   # 320
nameFile = ""
configFile = ""
weightsFile = ""

# Threshold Parametreleri
threshold_MNS = 0               # 0.2
threshold_CONF = 0              # 0.5

# Çizim için değişkenler
colorRectangle = (0,255,255)
colorCircle = (0,0,255)
colorText = (0,255,255)
font = font = cv.FONT_HERSHEY_COMPLEX_SMALL

# .names dosyasının okunması
nameOfClasses = []
with open(nameFile, "r") as f:
    nameOfClasses = [line.strip() for line in f.readlines()]


# Modelin oluşturulması ve ayarlanması
model = cv.dnn.readNetFromDarknet(configFile, weightsFile)
model.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA) # OPENCV
model.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)   # CPU

# Çıktı katmanlarının tespiti
"""
Bu işlemi döngü içerisinde yapmadım çünkü çıktı katmanları değişkenlik gösteren, 
yerleri değişen nesneler değillerdir. Bu yüzden her döngüde yeniden bu katmanları
tespit etmek gereksiz iş yükü olacaktır.
"""
layerNames = model.getLayerNames()
outputLayerNames = [(layerNames[i - 1]) for i in model.getUnconnectedOutLayers()]

"""
Modelin çıktılarının tespit edilip aralarında ayıklama yapan kodları bir fonksiyon
haline getirdim. Aksi şekilde döngü içerisinde kullanırsam eğer alabileceğim FPS 
performansını düşürmüş olacağım.
"""

def detection(img, outputLayers):
    imgHeight, imgWidth = img.shape[:2]

    objectsIds = []
    boundingBoxes = []
    confidenceRates = []

    for layer in outputLayers:
        for bbox in layer:

            detectionScores = bbox[5:]
            objectId = np.argmax(detectionScores)
            confidence = detectionScores[objectId]

            if confidence > 0.7:
                w = int(bbox[2] * imgWidth)
                h = int(bbox[3] * imgHeight)
                x = int((bbox[0] * imgWidth) - w / 2)
                y = int((bbox[1] * imgHeight) - h / 2)

                objectsIds.append(objectId)
                boundingBoxes.append([x, y, w, h])
                confidenceRates.append(float(confidence))

    indexes = cv.dnn.NMSBoxes(boundingBoxes, confidenceRates, threshold_CONF, threshold_MNS)

    for i in indexes:
        box = boundingBoxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cx, cy = int(x + w / 2), int(y + h / 2)
        cv.circle(img, (cx, cy), 3, colorCircle, -1)
        cv.rectangle(img, (x, y), (x + w, y + h), colorRectangle, 1)
        cv.putText(img, "{}, %{}".format(nameOfClasses[objectsIds[i]].upper(),
                                             int(confidenceRates[i] * 100)), (x, y - 10), font, 1, colorText, 1)

# Döngünün ile kamera değerlerinin okuma
while cap.isOpened():
    _, frame = cap.read()

    # Resimi hazırlama ve modele gönderme
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inputSize, inputSize), [0, 0, 0], 1, crop=False)
    model.setInput(blob)

    # Daha önceden bulduğumuz çıktı katmanlarından sonuçların çekilmesi
    outputLayers = model.forward(outputLayerNames)

    # Elde edilen stespit sonuçlarının ayıklanması, seçilmesi ve çizlmesi
    detection(frame, outputLayers)

    # Sonuçların ekranda gösterilmesi
    cv.imshow("FRAME", frame)
    if(cv.waitKey(1) == ord("q")):
        break

cap.release()
cv.destroyAllWindows()