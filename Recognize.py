import numpy as np
import cv2
import keras_ocr


def center(width, height, x, y, w, h):
    _w = int(w * width)
    _h = int(h * height)
    _x = int(x * width - _w / 2)
    _y = int(y * height - _h / 2)
    return _x, _y, _w, _h


def recognize(img):
    height, width, channels = img.shape
    boxes, confidences = [], []

    net = cv2.dnn.readNet("y4plate.weights", "y4plate.cfg")

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    net.setInput(cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False))
    outs = net.forward(output_layers)

    for out in outs:

        for detection in out:
            scores = detection[5:]
            confidence = scores[np.argmax(scores)]

            if confidence > 0.25:
                boxes.append(center(width, height, detection[0], detection[1], detection[2], detection[3]))
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    text, new_results, t = '', [], {}

    for i in range(len(boxes)):

        if i in indexes:
            x, y, w, h = boxes[i]
            plate = img[y:y + h, x:x + 2 * w]

            plate = cv2.resize(plate, (0, 0), fx=3, fy=3)
            results = keras_ocr.pipeline.Pipeline().recognize([plate])

            for r in range(len(results[0])):
                t[r] = results[0][r][1][0][0]

            s = sorted(t.items(), key=lambda xx: xx[1])

            for pr in range(len(results[0])):
                new_results.append(results[0][s[pr][0]])

            for p in range(len(results[0])):
                text += new_results[p][0]

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, text.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

            cv2.imshow('plate', plate)

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread("car1.jpg")
recognize(img)
