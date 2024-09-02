YOLOv9 Object Detection with DeepSORT Tracking(ID + Trails)

## Project Overview
This project aims to develop a person detection and tracking system to identify and track children with Autism Spectrum Disorder (ASD) and therapists in videos. The system is designed to assign unique IDs to individuals, track their movements, and handle cases of re-entry and occlusion. The ultimate goal is to analyze behaviors, emotions, and engagement levels to enhance treatment plans.

## Installation
1. Clone the repository:

git clone https://github.com/your-repo/your-project.git cd your-project

2. Install the required dependencies:
pip install -r requirements.txt

3. Ensure you have Python 3.8+ and a compatible GPU (if available).

4. Download the DeepSORT Files From The Google Drive 
gdown "https://drive.google.com/uc?id=11ZSZcG-bcbueXZC3rN08CM0qqX3eiHxf&confirm=t"
- After downloading the DeepSORT Zip file from the drive, unzip it. 

## Model Training (Optional)
The YOLOv9 model(saved as best.pt in my directory) was fine-tuned on a custom dataset containing images of adults and children. The training was done using the following steps:

- Dataset preparation: https://universe.roboflow.com/idan-kideckel-67kqi/children-and-adults/dataset/1

- Training script: https://colab.research.google.com/drive/1tj6v4qaqI2ETg_z0Ezopg-xHv2AIdLvP?usp=drive_link

- Hyperparameters: 
```
lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.05  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.001  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.1  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
dfl: 0.7  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
dfl: 1.7  # dfl loss gain
iou_t: 0.20  # IoU training threshold
anchor_t: 5.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.9  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.2  # image mixup (probability)
copy_paste: 0.3  # segment copy-paste (probability)
```

#for detection and tracking
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0

#for WebCam
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 0 --device 0

#for External Camera
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 1 --device 0

#For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source "your IP Camera Stream URL" --device 0

#for specific class (person)
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --classes 0


- Output file will be created in the ```working-dir/runs/detect/obj-tracking``` with original filename


