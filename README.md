# Droid-Quality-Monitor
A computer vision powered quality monitor and dashboard

![demo_video](https://github.com/user-attachments/assets/67d94c2e-d4d4-4790-ac3c-de24dd8b9aa4)

This app runs a custom-trained YOLO11 model to detect discolorations in lego droid pieces. Becuase there are so many possible discolorations, and I did not have many spare pieces on hand, much of the data (over 95%) used for model training was synthetically generated. Data was generated using Nvidia Omniverse, where I ran 5 simulations to create synthetic data, where a piece of the droid body was randomized in color from a distribution of colors which would not appear similar to the correct color. The distribution is defined in the custom omniverse extension omni.graph.henry. The app is written using pyqt6 and creates a p-chart to monitor the proportion of droids with at least one defect that the camera can see. 
