from ultralytics import YOLO
import cv2 as cv


model = YOLO('runs/detect/train6/weights/best.pt')


cap = cv.VideoCapture(0)



while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Display the resulting frame
    result = model.predict(frame, conf=.5)

    cv.imshow('test', result[0].plot())


    # Break the loop if 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv.destroyAllWindows()