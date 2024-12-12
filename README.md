# Object Detection with YOLO

This project demonstrates object detection using the YOLO model with OpenCV and PyTorch. The project processes both images and videos, saving the results in specified output directories.

## Project Structure

```
/D:/Perso/Project_IA/Object_Detection_Computer_Vison/
│
├── main.py
├── video_detection.py
├── utils.py
├── yolo/
│   ├── yolov3.cfg
│   ├── yolov3.weights
│   └── coco.names
├── data/
│   ├── images/
│   └── videos/
└── output/
    ├── images/
    └── videos/
```

## Setup

1. Clone the repository.
2. Download the YOLO configuration, weights, and class names files and place them in the `yolo/` directory.
3. Install the required Python packages:
   ```sh
   pip install opencv-python-headless numpy torch matplotlib
   ```

## Usage

### Detect Objects in Images

1. Place your images in the `data/images` directory.
2. Run the `main.py` script:
   ```sh
   python main.py
   ```
3. Processed images will be saved in the `output/images` directory.

### Detect Objects in Videos

1. Place your videos in the `data/videos` directory.
2. Run the `video_detection.py` script:
   ```sh
   python video_detection.py
   ```
3. Processed videos will be saved in the `output/videos` directory.

## Example Results

### Image Detection

Original Image:
![Original Image](data/images/sample_image.jpg)

Processed Image:
![Processed Image](output/images/sample_image.jpg)

### Video Detection

Original Video:
![Original Video](data/videos/sample_video.mp4)

Processed Video:
![Processed Video](output/videos/sample_video.mp4)

## License

This project is licensed under the MIT License.
