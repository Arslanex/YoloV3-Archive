import cv2 as cv
import numpy as np

# Eğitim sonucu oluşan .cfg ve .weights dosyalarını kullanarak modeli başlattım
net = cv.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')

# Çıktı katmanlarını diğer katmanlardan ayıkladım
output_layers_names = net.getUnconnectedOutLayersNames()

# Drive üzerinden ibdirdiğim isim dosyasını listeye çektiö
classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

cap = cv.VideoCapture("vid1.mp4")

while cap.isOpened():
    _, frame = cap.read()
    height, width, _ = frame.shape

    # Videodan gelen kareleri model için uygun hale getirdim ve modele gönderdim
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Çıktıları aldım
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            cv.rectangle(frame, (x, y), (x + w, y + h), (255,255,0), 2)
            cv.putText(frame, label + " " + confidence, (x, y + 20), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv.imshow('Image', frame)
    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
