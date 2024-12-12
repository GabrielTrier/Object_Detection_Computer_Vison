import cv2
import numpy as np

# Chargement des classes COCO
def load_classes(file_path):
    with open(file_path, "r") as f:
        return f.read().strip().split("\n")

# Charger le modèle YOLOv3
def load_model(cfg_path, weights_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# Prétraitement des images
def preprocess_image(image, input_size):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, input_size, swapRB=True, crop=False)
    return blob

# Post-traitement des prédictions
def postprocess_predictions(layer_outputs, width, height, confidence_threshold, nms_threshold):
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    return idxs, boxes, confidences, class_ids
