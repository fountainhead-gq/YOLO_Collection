import colorsys
import imghdr
import os
import sys
import cv2
import random
from keras import backend as K

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def preprocess_image(img_path, model_image_size):
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))
        #sys.stdout.flush()
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

        
def detect_video(image_path, output_path, model_image_size, yolo_model, class_names, yolo_outputs): 
    from keras import backend as K
    from keras.models import load_model
    from yad2k.models.keras_yolo import yolo_head, yolo_eval
    from PIL import Image
    input_image_shape = K.placeholder(shape=(2, ))
    sess = K.get_session()
    
    video_in = cv2.VideoCapture(image_path)
    #width = video_in.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float
    #height = video_in.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) # float
    width, height = int(video_in.get(3)), int(video_in.get(4))
    FPS = video_in.get(5)
    
    video_out = cv2.VideoWriter()
    video_out.open(output_path, cv2.VideoWriter_fourcc(*'DIVX'), FPS, (width, height))
    
    width = np.array(width, dtype=float)
    height = np.array(height, dtype=float)
    image_shape = (height, width)
    
    while video_in.isOpened():
        ret, data = video_in.read()
        if ret==False:
            break
        video_array = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
        image = Image.fromarray(video_array,mode='RGB')
        resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)   # Add batch dimension.
        
        boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
        out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes],
                                                      feed_dict={yolo_model.input: image_data,
                                                                 input_image_shape: [image.size[1], image.size[0]],
                                                                 K.learning_phase(): 0})
        colors = generate_colors(class_names)
        draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
        video_out.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        
    sess.close()
    video_in.release()
    video_out.release()
    print("detect Done")        
       