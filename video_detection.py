import os
import cv2
import torch
import numpy as np
from utils import load_classes, load_model, preprocess_image, postprocess_predictions

# Détection sur une vidéo
def detect_objects_video(video_path, net, classes, input_size, confidence_threshold, nms_threshold, output_dir):
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    output_path = os.path.join(output_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        (H, W) = frame.shape[:2]
        blob = preprocess_image(frame, input_size)
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

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y, w, h) = boxes[i]
                color = [int(c) for c in np.random.randint(0, 255, size=(3,), dtype="uint8")]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, text, (x, y - 5), font, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")

# Configurations principales
if __name__ == "__main__":
    # Chemins des fichiers YOLO
    cfg_path = "yolo/yolov3.cfg"
    weights_path = "yolo/yolov3.weights"
    classes_path = "yolo/coco.names"

    # Chargement des classes et du modèle
    classes = load_classes(classes_path)
    net = load_model(cfg_path, weights_path)

    # Détection sur un dossier de vidéos
    video_dir = "data/videos"
    output_dir = "output/videos"
    os.makedirs(output_dir, exist_ok=True)
    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        detect_objects_video(video_path, net, classes, (416, 416), 0.5, 0.4, output_dir)
