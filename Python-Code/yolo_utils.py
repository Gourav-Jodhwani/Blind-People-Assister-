import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
import pyttsx3

engine = pyttsx3.init()

def show_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)

def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    # If there are any detections
    cv.line(img, (640//3, 0), (640//3, 480), (0,0,0), 2)
    cv.line(img, ((640*2)//3, 0), ((640*2)//3, 480), (0,0,0), 2)
    cv.line(img, (0,480//3), (640,480//3), (0,0,0), 2)
    cv.line(img, (0,(480*2)//3), (640,(480*2)//3), (0,0,0), 2)
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            
            
            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)

            text_horiz = "{}: {:4f}".format(labels[classids[i]], confidences[i])
            text_vert = "{}: {:4f}".format(labels[classids[i]], confidences[i])


            topLeft,topRight=x,x+w
            vert_top,vert_bottom=y,y+h
            left=640//3
            right=(640*2)//3
            top=480//3
            bottom=(480*2)//3

            if( topLeft<left and topRight<=left):
                text_horiz="Left"
            if(topLeft>=left and topRight<=right):
                text_horiz="Middle"
            if(topLeft>=right and topRight>=right):
                text_horiz="Right"
            if(topLeft<left and topRight>left and topRight<=right):
                text_horiz="Left and Middle"
            if(topLeft>=left and topLeft<right and topRight>=right):
                text_horiz="Middle and right"
            if( topLeft<=left and topRight>=right):
                text_horiz="Left and Middle and Right"

            if( vert_top<top and vert_bottom<=top):
                text_vert="Top"
            if(vert_top>=top and vert_bottom<=bottom):
                text_vert="Middle"
            if(vert_top>=bottom and vert_bottom>=bottom):
                text_vert="Bottom"
            if(vert_top<top and vert_bottom>top and vert_bottom<=bottom):
                text_vert="Top and Middle"
            if(vert_top>=top and vert_top<bottom and vert_bottom>=bottom):
                text_vert="Middle and Bottom"
            if( vert_top<=top and vert_bottom>=bottom):
                text_vert="Top and Middle and Bottom"
            

            engine.say(labels[classids[i]]+" "+text_vert+" "+text_horiz)
            engine.runAndWait()

            cv.putText(img, labels[classids[i]]+" "+text_vert+" "+text_horiz, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            #print (detection)
            #a = input('GO!')
            
            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            
            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf:
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids

def infer_image(net, layer_names, height, width, img, colors, labels, FLAGS, 
            boxes=None, confidences=None, classids=None, idxs=None, infer=True):
    
    if infer:
        # Contructing a blob from the input image
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), 
                        swapRB=True, crop=False)

        # Perform a forward pass of the YOLO object detector
        net.setInput(blob)

        # Getting the outputs from the output layers
        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()

        if FLAGS.show_time:
            print ("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

        
        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, FLAGS.confidence)
        
        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, FLAGS.confidence, FLAGS.threshold)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'
        
    # Draw labels and boxes on the image
    img = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels)

    return img, boxes, confidences, classids, idxs
