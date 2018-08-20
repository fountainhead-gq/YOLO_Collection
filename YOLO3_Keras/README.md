# yolo3-keras



## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
```
wget https://pjreddie.com/media/files/yolov3.weights
```
2. Convert the Darknet YOLO model to a Keras model.
```
python convert.py yolov3.cfg yolov3.weights model_data/yolo3.h5
```
3. Run YOLO detection.

```
# detect image
from yolo3_predict import YOLO
yolo = YOLO()
...
save_image = yolo.detect_image(input_image)

# detect video
...
yolo.detect_video(video_path, output_path)
```

## Training

1. Data preparation

    download  PASCAL VOC dataset from [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)  
    run `python voc_annotation.py`  


2. Make sure you have the pretrained weights `model_data/yolo3.h5`  


3. Modify `yolo3_train.py` and start training.  
    `python yolo3_train.py`  



