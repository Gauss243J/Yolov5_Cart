import torch
import cv2
from matplotlib import pyplot as plt
import pandas as pd

# Load the YOLOv5 model
model_path = '/home/pi/last.pt'  # Update to the correct model path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image from BGR to RGB (YOLO expects RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make predictions
    results = model(img_rgb)

    # Display the results
    results_img = results.render()[0]  # Get the annotated image
    cv2.imshow('YOLOv5 Detection', results_img)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Extract data from results
    df = results.pandas().xyxy[0]  # Get predictions as a pandas DataFrame
    print(df)

    # Count the number of detected objects by class
    object_counts = df['name'].value_counts()
    for object_name, count in object_counts.items():
        print(f'{count} {object_name}(s) detected')

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
