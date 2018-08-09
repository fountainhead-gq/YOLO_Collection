## YOLOv1 (tensorflow)

Tensorflow implementation of [YOLO](https://arxiv.org/pdf/1506.02640.pdf), including training and test .

### Installation

1. Download Pascal VOC dataset, and download YOLO_small weight file and put it in data/weight

2. detect
```python
yolo = YOLO_TF()
weight_file = os.path.join(r'data/weights','YOLO_small.ckpt')
detector = YOLO_Detector(yolo, weight_file)

img_name = 'test/cat.jpg'
detector.image_detector(img_name)
```

### Requirements
1. Tensorflow  
2. OpenCV
