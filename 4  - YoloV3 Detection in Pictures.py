import cv2 as cv
import numpy as np

target_img = cv.imread("")
imgHeight, imgWidth = target_img.shape [:2]

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


# Resimin hazırlanması ve modele gönderilmesi
blob = cv.dnn.blobFromImage(target_img, 1 / 255, (inputSize, inputSize), [0, 0, 0], 1, crop=False)
model.setInput(blob)

# Çıktı katmanlarının bulunup çıktı verilerinin toplanması
model.setInput(blob)
layerNames = model.getLayerNames()
outputLayerNames = [(layerNames[i - 1]) for i in model.getUnconnectedOutLayers()]
outputLayers =  model.forward(outputLayerNames)

# Çıktıların kontrol edilip işe yarayanların toplanması
## Depolama için listeler
objectsIds = []
boundingBoxes = []
confidenceRates = []

## Katmanlar içinden her katmanın sahip olduğu her kutunnun incelenme işlemi
for layer in outputLayers:
    for bbox in layer:

        detectionScores = bbox[5:]
        objectId = np.argmax(detectionScores)
        confidence = detectionScores[objectId]

        if confidence > 0.7:

            w = int(bbox[2] * imgWidth)
            h = int(bbox[3] * imgHeight)
            x = int((bbox[0] * imgWidth)-w/2)
            y = int((bbox[1] * imgHeight)-h/2)

            objectsIds.append(objectId)
            boundingBoxes.append([x, y, w, h])
            confidenceRates.append(float(confidence))

# Üst üste binen kutuların ayıklanması
indexes = cv.dnn.NMSBoxes(boundingBoxes, confidenceRates, threshold_CONF, threshold_MNS)

# Kalan kutuların çizim işlemi
for i in indexes:
    box = boundingBoxes[i]
    x, y, w, h = box[0], box[1], box[2], box[3]
    cx, cy = int(x + w / 2), int(y + h / 2)

    cv.circle(target_img, (cx, cy), 3, colorCircle, -1)
    cv.rectangle(target_img, (x, y), (x + w, y + h), colorRectangle, 1)

# Sonuçların ekranda gösterilmesi
cv.imshow('IMAGE', img)
cv.waitKey(0)