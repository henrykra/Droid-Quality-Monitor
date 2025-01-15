import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from ultralytics import YOLO
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


def calc_control_lims(p_data):
    pbar = np.mean(p_data)

    se = 3 * np.sqrt((pbar * (1-pbar)) / len(p_data))

    return max(0, pbar - se), min(1, pbar + se)


def draw_boxes(result, show_defects):
    """
    Function that draws a single frame of bounding boxes given a result
    
    Args:
        result: single ultralytics result object
        class_inputs: list of user selections
        
    Returns: 
        Image array with drawn on bounding boxes (BGR)"""
    
    # tensor in form ([x1, y1, x2, y2, conf, class], []...)
    data = result.boxes.data

    img = result.orig_img
    line_thickness = 2

    for box in data:
        pred_class = int(box[-1])

        if not show_defects and pred_class > 0:
            continue

        # check if the predicted class has a value of true in class inputs
        # meaning that it should be marked red for removal
        if pred_class > 0: # if it is a defect class TODO: change condition to be more intuitive
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3])) # coordinates
        # draw rectangle
        cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

    return img


class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.show_defects = False
        self.results = None
        self.max_id = 0 # For droid tracking
        self.sample_num = 0

        self.setWindowTitle("Live Object Detection")
        self.setGeometry(100, 100, 800, 900)

        # Create a label to display the webcam feed
        self.label = QLabel(self)
        #self.label.setFixedSize(800, 600)

        # Create a button to change the box color
        self.button = QPushButton("Show Defects", self)
        self.button.clicked.connect(self.detect_defects)

        self.sample = QPushButton("Collect Sample Data", self)
        self.sample.clicked.connect(self.collect_sample)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("P-Chart")
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("p_hat")
        self.x_data = []
        self.p_data = []
        self.mean = self.ax.axhline(y=0, xmin=0, xmax=0, ls='--', c='r', alpha=.7)
        self.ucl = self.ax.axhline(y=0, xmin=0, xmax=0, ls='-', c='g', alpha=.5)
        self.lcl = self.ax.axhline(y=0, xmin=0, xmax=0, ls='-', c='g', alpha=.5)
        self.ax.set_ylim(0, 1)
        self.plot, = self.ax.plot(self.x_data, self.p_data, marker='o')


        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.label)
        layout.addWidget(self.canvas)
        layout.addWidget(self.sample)
        self.setLayout(layout)

        # Initialize the YOLO model
        self.model = YOLO("runs/detect/train6/weights/best.pt")  # Load a pre-trained YOLO model

        # Set up the webcam
        self.capture = cv2.VideoCapture(0)  # 0 is the default webcam

        if not self.capture.isOpened():
            print("Error: Could not open webcam.")
            sys.exit()

        # Set up a timer to update the frame periodically
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms


    def collect_sample(self):
        # Check detection boxes for defective samples
        droids_coords = []
        defects = 0 
        good = 0
        

        # Get xmin, xmax, ymin and ymax for all droid detections
        for box in self.results[0].boxes:
            if box.cls[0] == 0: # droid class
                xmin, ymin, xmax, ymax = box.xyxyn[0] # normaized coordinates
                droids_coords.append((xmin, xmax, ymin, ymax))

        # look for defect boxes and check if they are inside droids
        good = len(droids_coords)
        if good == 0:
            print("No Droids Detected")
            return
            

        for xmin, xmax, ymin, ymax in droids_coords:
            for box in self.results[0].boxes:
                if box.cls[0] > 0: # defect box
                    coords = box.xywhn[0]
                    x, y = (coords[0], coords[1])# normalized center coords
                    if x > xmin and x < xmax and y > ymin and y < ymax:
                        defects += 1
                        break


        good -= defects

        assert good >= 0

        p = defects / (defects + good)
        assert p <= 1
        assert p >= 0

        self.sample_num += 1
        # update control chart

        # TODO: better sample number logic
        self.x_data.append(self.sample_num)
        self.p_data.append(p)



        self.plot.set_data(self.x_data, self.p_data)

        # pbar line
        self.mean.set_ydata([np.mean(self.p_data), np.mean(self.p_data)])
        self.mean.set_xdata([0, max(self.x_data)])

        # control limits
        lcl, ucl = calc_control_lims(self.p_data)
        self.lcl.set_ydata([lcl, lcl])
        self.ucl.set_ydata([ucl, ucl])

        self.lcl.set_xdata([0, max(self.x_data)])
        self.ucl.set_xdata([0, max(self.x_data)])

        self.ax.relim()
        self.ax.autoscale_view()
        

        self.canvas.draw()
  


    def detect_defects(self):
        # Toggle whether to draw defect boxes
        self.show_defects = not self.show_defects

        # Toggle box border to show when clicked
        # if self.show_defects:
        #     self.button.setStyleSheet(
        #         "border: 2px solid red"
        #     )
        # else:
        #     self.button.setStyleSheet(
        #         "border: 2px solid black"
        #     )


    def update_frame(self):

        ret, frame = self.capture.read()

        if not ret:
            print("Error: Could not read frame from webcam.")
            return

        # Perform object detection using YOLO
        self.results = self.model.track(frame, conf=.4, verbose=False, persist=True)

        # Check if a new droid is found
        if self.results[0].boxes.is_track:
            classes = self.results[0].boxes.cls.cpu().numpy()
            ids = self.results[0].boxes.id.cpu().numpy()

            # Find the ids for droids in the frame (class 0)
            droid_ids = ids[np.where(classes == 0)]
            droid_ids.sort()

            # Search ids within the frame, and check if any are greater than the previous greatest
            for id in droid_ids:
                if id > self.max_id:
                    print(f'New Droid Found: {id}')
                    self.max_id = id

        

        annot_frame = draw_boxes(self.results[0], self.show_defects)

        frame_rgb = cv2.cvtColor(annot_frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a QImage for display
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        qimg = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Display the QImage on the QLabel
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        # Release the webcam when the app is closed
        self.capture.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec())
