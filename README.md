# Anomaly Detection using YOLOv8
### Real world and real time automated Anomaly Detection project using YOLOv8

## Features
  - Real time detection using visual data from CCTV/surveillance cameras.
  - Detects variety of anomalies like fire, road accidents etc, provided the dataset.
  - Saves only the anomalous part of the surveillance video ensuring effecient storage.
  - Pop-up message in system as anomaly is detected 
  - Supports real time viewing

## Tools
  - Google Colab (Training)
  - VS Code (RUnning final program)

## Training YOLOv8 model
  - Open Colab and follow the steps:
  1. It is better to mount the drive to Colab as it will be useful for repetitive testing and training:
  ```
  from google.colab import drive
  drive.mount('/content/drive')
  ```
  2. Install `Ultralytics`:
  ```
  !pip install ultralytics
  ```
  3. Create datasets using sources like Kaggle and Roboflow.
  4. Upload the datasets into a Folder in the drive, in this case we can call the folder 'YOLO'.
  5. Split the datasets into three more folders as 70% to 80% of the data for training, 10% to 15% for validation, and 10% to 20% for testing.
  The directory structure will be like:
  YOLO/
    ├── data.yaml         
    ├── train            
    ├── valid           
    └── test    
  6. Install `YOLOv8`:
  - If you have used roboflow then the 'data.yaml' file will come along with it:
  ```
  from ultralytics import YOLO
  import os
  !yolo mode=checks
  ```
  - If no 'data.yaml' is found then  install YOLO using:
  ```
  !git clone https://github.com/ultralytics/ultralytics
  import os
  !yolo mode=checks
  ```
  And then setup the yaml as:
  ```
  %cd /content/ultralytics/ultralytics
  with open('/content/drive/MyDrive/YOLO/data.yaml','x') as f:
    f.write("""train: /content/drive/MyDrive/YOLO/train
  val: /content/drive/MyDrive/YOLO/valid
  test: /content/drive/MyDrive/YOLO/test
  nc: 4
  names: ['Accident','Fight','Fire','Smoke']""")
  ```
  Here `nc` is the number of classes i.e, Accident,Fire,Smoke,Fight.
  7. Training:
  ```
  !yolo task=detect mode=train model=[input_model_name_here] data='/content/YOLO/data.yaml' imgsz=640 batch=3 epochs=100
  ```
  Tweak `imgsz`, `batch` and `epochs` according to your will.
  8. Validate:
  ```
  !yolo task=detect mode=val model='/content/drive/MyDrive/modeltesting/[input_model_name_here]' data='/content/drive/MyDrive/YOLO/data.yaml'
  ```
  9. Test:
  ```
  !yolo task=detect mode=predict model=/content/drive/MyDrive/modeltesting/[input_model_name_here] conf=0.6 imgsz=640 save=True source=/content/drive/MyDrive/[testing_video_path]
  ```
  Tweak `imgsz` and `conf` according to your will.

## Detection program
  1. Prerequisites:
  ```
  !pip install ultralytics
  !pip install requests
  ``` 
  ```
  import torch
  import numpy as np
  import cv2
  from ultralytics import YOLO

  import threading
  import requests
  import subprocess
  import keyboard
  ```
  2. Pop-up vbs script
  ```
  Dim result
  Dim message

  ' Check if a command-line argument is provided
  If WScript.Arguments.Count > 0 Then
      ' Retrieve the message from the command-line argument
      message = WScript.Arguments(0)
  Else
      ' Default message if no argument is provided
      message = "Click OK to redirect to the surveillance feed."
  End If

  ' Display the message in a pop-up window
  result = MsgBox(message, vbOKCancel + vbInformation, "QBot Alert")

  If result = vbOK Then
      ' Redirect to the desired webpage
      Set objShell = CreateObject("WScript.Shell")
      objShell.Run "[website_link]"
  End If
  ```
  3. Main module:
  ```
  class AnomalyDetection:
      def __init__(self, capture_source, camera_name):
          self.capture_source = capture_source
          self.camera_name = camera_name
          self.msg_sent = False
          self.frames=[]
          self.no_detct = 0
          i=-1
          # Device info
          self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
          # Model info
          self.model = YOLO(r"[path_to_the_model.pt]")

      def send_msg(self, msg):
          print(msg)
          subprocess.call(['cscript', 'popup.vbs', f"{self.camera_name} - Anomaly Detected: {msg}"], shell=True)
          current_datetime = datetime.datetime.now().strftime(r"%Y-%m-%d%H-%M-%S")
          out_path2='C:/Users/shree/OneDrive/Desktop/Yolo2/svd/list.txt'
          with open(out_path2, "a") as f:
                  f.write(f"Camera 1 {current_datetime} {','} {msg}\n")

      def frames_to_video(self, out_path, fps=2):
      # Determine the shape of the frames
          size = (self.frames[0].shape[1], self.frames[0].shape[0])
          # Define the codec and create VideoWriter object
          fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choose the codec (here, MP4V)
          out = cv2.VideoWriter(out_path, fourcc, fps, size)

          # Write each frame to the video
          for frame in self.frames:
              out.write(frame)

          # Release the VideoWriter object
          out.release()
          self.frames=[]
          print(f"Video saved to: {out_path}")


      def predict(self, im0):
          clas=-1

          results = self.model.predict(source=im0, conf=0.6)
          anomaly = ["Accident", "Fight", "Fire", "Smoke"]
          for r in results:
              clss = r.boxes.cls.cpu().numpy().astype(int)
              con = r.boxes.conf.cpu().numpy().astype(float)
              print(con)
              print(clss)
              if clss == 0 and con >=0.5 :
                clas=clss
                print("in acc")
              elif clss == 1 and con >=0.75 :
                clas=clss
              elif clss == 2 and con >=0.7 :
                clas=clss
                print("in fire")
              elif clss == 3 and con >=0.7 :
                clas=clss


          if clas==-1:
              self.no_detct= self.no_detct+1

          if  clas >=0:
              print(clas)
              self.no_detct=0
              self.frames.append(im0)
              if not self.msg_sent:
                  for cl in clas:
                      self.send_msg(anomaly[cl])
                  self.msg_sent = True
          else:
              self.msg_sent = False
          if self.no_detct > 10 and len(self.frames)>0:
              self.no_detct = 0
              current_datetime = datetime.datetime.now().strftime(r"%Y-%m-%d%H-%M-%S")
              out_path = rf'[output_path_to_save_the_anomalous_video_part]\{current_datetime}.mp4'
              self.frames_to_video(out_path)



      def _call_(self):
          cap = cv2.VideoCapture(self.capture_source)
          assert cap.isOpened()
          cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
          cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
          while True:
              ret, im0 = cap.read()
              assert ret
              self.predict(im0)
              if cv2.waitKey(5) & 0xFF == 27 or keyboard.is_pressed('esc'):
                  break
          cap.release()
          cv2.destroyAllWindows()

  # Instantiate anomaly detection for each camera
  detector1 = AnomalyDetection(capture_source=0, camera_name='Camera 1') # '0' indicates that the webcam is used.
  #detector2 = AnomalyDetection(capture_source='[place_your_camera_ip_here]', camera_name='Camera 2')

  # Start anomaly detection for each camera in separate threads
  thread1 = threading.Thread(target=detector1._call_)
  #thread2 = threading.Thread(target=detector2._call_)

  # Start both threads
  thread1.start()
  #thread2.start()

  # Wait for both threads to finish
  thread1.join()
  #thread2.join()
  ```

## Disclaimer
> Note: If you are using an newer version of YOLO, refer to the [documentation] (https://www.ultralytics.com/yolo) for the version for the inbuilt YOLO methods and functions.

This project is for educational purposes only. Use responsibly and at your own risk. Any misuse will not be the responsibility of the developers nor the distributors.