import tensorflow_hub as hub
import cv2
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os
import shutil

width = 1028
height = 1028


count = 0

# Loading model directly from TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")

# Loading csv with labels of classes
labels = pd.read_csv('../labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']


    
# cap = cv2.VideoCapture('balls/0BSAH0I3BHLJ.mp4')
# length = int(cap.get(cv2.CAP_PROP_FPS))
# print( length )
# ret, frame = cap.read()
# exit()

for file in os.scandir('../strikes'):
    print(count)
    count+=1

    cap = cv2.VideoCapture(file.path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    out = cv2.VideoWriter(file.path.split('/')[2].split('.')[0] + '.mp4' , fourcc, 20.0, (frame_width,frame_height))

    frame_count = 0


    while cap.isOpened():
        
        ret, frame = cap.read()

        frame_count +=1
        # cv2.imshow('', frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if ret:
            #Load image by Opencv2
            #img = cv2.imread('screenshot.png')
            #Resize to respect the input_shape
            # inp = cv2.resize(frame, (width , height ))



            #Convert img to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # COnverting to uint8
            rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

            #Add dims to rgb_tensor
            rgb_tensor = tf.expand_dims(rgb_tensor , 0)

            boxes, scores, classes, num_detections = detector(rgb_tensor)
            # Processing outputs
            pred_labels = classes.numpy().astype('int')[0] 
            if numpy.count_nonzero(pred_labels == 37) >=2:
                if frame_count <=10:
                    os.remove(file.path.split('/')[2].split('.')[0] + '.mp4' )
                cap.release()
                #out.release()
                break
        else:
            break

cv2.destroyAllWindows()
    

# video = numpy.stack(frames, axis=0)

# first_frame = video[0]

# middle_frame = video[60]

# diff = middle_frame-first_frame


# print(diff)

# print(diff.shape)





# pred_boxes = boxes.numpy()[0].astype('int')
# pred_scores = scores.numpy()[0]

# print(pred_labels)

# # Putting the boxes and labels on the image
# for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
#     if score < 0.5:
#         continue

#     score_txt = f'{100 * round(score)}%'
#     img_boxes = cv2.rectangle(rgb,(xmin, ymax),(xmax, ymin),(0,255,0),2)      
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(img_boxes, label,(xmin, ymax-10), font, 1.5, (255,0,0), 2, cv2.LINE_AA)
#     cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 1.5, (255,0,0), 2, cv2.LINE_AA)

# plt.imshow(img_boxes)

# cv2.imwrite('test.jpeg', img_boxes)

# print(type(img_boxes))



# print('done')