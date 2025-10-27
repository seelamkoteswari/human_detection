# Human detection
import cv2
import torch 


# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5','yolov5s' , pretrained=True)

#Load the webcam feed or video file
video_capture = cv2.VideoCapture(0) #use 0 for webcam; replace with file path for video

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Run the YOLO model on the frame
    results = model(rgb_frame)
    # Extract detections and filter by class(e.g. ,person)
    detection = results.pred[0]
    for *box, conf, cls in detection.cpu().numpy():
        if int(cls) == 0:  #  Class 0 corresponds to 'person'
            x1,y1,x2,y2,= map(int, box)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)  #Draw bounding box
            cv2.putText(frame, f'person:{conf:.2f}', (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)


            #Display the frame
            cv2.imshow('Human detection', frame)
             #Braek loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
#realse resources 
video_capture.release()
cv2.destroyALLWindows()


