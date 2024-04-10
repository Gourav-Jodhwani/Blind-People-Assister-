import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "http://192.168.1.4:8080/video"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame,stream_buffer=True,half=True)
        for result in results:
            for box in result.boxes.cpu().numpy():
                r = box.xyxy[0].astype(int)
                cv2.rectangle(frame, r[:2], r[2:], (255, 255, 255), 2)
                

        # Visualize the results on the frame
        # annotated_frame = results[0].plot()

        # Display the annotated frame
        # cv2.imshow("YOLOv8 Inference", annotated_frame)
        cv2.imshow("YOLOv8 Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
