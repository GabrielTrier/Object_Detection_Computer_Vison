import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_classes, load_model, preprocess_image, postprocess_predictions

# Détection sur une image
def detect_objects_image(image_path, net, classes, input_size, confidence_threshold, nms_threshold, output_dir):
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]

    blob = preprocess_image(image, input_size)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_out_layers, np.ndarray):
        unconnected_out_layers = unconnected_out_layers.flatten()
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
    layer_outputs = net.forward(output_layers)

    idxs, boxes, confidences, class_ids = postprocess_predictions(
        layer_outputs, W, H, confidence_threshold, nms_threshold
    )

    for i in idxs.flatten():
        (x, y, w, h) = boxes[i]
        color = [int(c) for c in np.random.randint(0, 255, size=(3,), dtype="uint8")]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, (x, y - 5), font, 0.5, color, 2)

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)

# Configurations principales
if __name__ == "__main__":
    # Chemins des fichiers YOLO
    cfg_path = "yolo/yolov3.cfg"
    weights_path = "yolo/yolov3.weights"
    classes_path = "yolo/coco.names"

    # Chargement des classes et du modèle
    classes = load_classes(classes_path)
    net = load_model(cfg_path, weights_path)

    # Détection sur un dossier d'images
    data_path = "data/images"
    output_dir = "output/images"
    os.makedirs(output_dir, exist_ok=True)
    for image_file in os.listdir(data_path):
        image_path = os.path.join(data_path, image_file)
        detect_objects_image(image_path, net, classes, (416, 416), 0.5, 0.4, output_dir)
