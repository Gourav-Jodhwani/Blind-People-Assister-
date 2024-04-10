# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

from yolo_utils import infer_image, show_image

# Import other dependencies
import cv2
import tensorflow as tf
import os
import numpy as np
import argparse
import subprocess

class CamApp(App):

    def build(self):
        # Main layout components 
        self.web_cam1 = Image(size_hint=(1,.8))
        self.web_cam2 = Image(size_hint=(1,.8))
        self.urlInput1=TextInput(hint_text="Enter URL here",size_hint=(1,0.1))
        self.urlInput2=TextInput(hint_text="Enter URL here",size_hint=(1,0.1))
        self.submit=Button(text="Submit",on_press=self.submit_url,size_hint=(1,0.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout2=GridLayout(cols=2)
        layout2.add_widget(self.web_cam1)
        layout2.add_widget(self.web_cam2)
        layout.add_widget(layout2)
        layout.add_widget(self.urlInput1)
        layout.add_widget(self.urlInput2)
        layout.add_widget(self.submit)

        # Setup video capture device
        self.url1=None
        self.url2=None
        self.capture1 = cv2.VideoCapture(self.url1)
        self.capture2 = cv2.VideoCapture(self.url2)
        self.count=0
        self.boxes1=None
        self.confidences1=None
        self.classids1=None 
        self.idxs1=None
        self.boxes2=None
        self.confidences2=None
        self.classids2=None 
        self.idxs2=None
        Clock.schedule_interval(self.update,1.0/30.0)
        
        return layout
    
    def submit_url(self,*args):
        self.url1=self.urlInput1.text
        self.url2=self.urlInput2.text


    def update(self, *args):
        if(self.url1 is None or self.url2 is None):return
        self.capture1 = cv2.VideoCapture(self.url1)
        self.capture2 = cv2.VideoCapture(self.url2)
        if (self.capture1 is None or not self.capture1.isOpened()):
            return
        if (self.capture2 is None or not self.capture2.isOpened()):
            return

        _, frame1 = self.capture1.read()
        height1, width1 = frame1.shape[:2]

        _, frame2 = self.capture2.read()
        height2, width2 = frame2.shape[:2]


        if self.count == 0:
            frame1, self.boxes1, self.confidences1, self.classids1, self.idxs1 = infer_image(net1, layer_names, \
                                height1, width1, frame1, colors, labels, FLAGS)
            frame2, self.boxes2, self.confidences2, self.classids2, self.idxs2 = infer_image(net2, layer_names, \
                                height2, width2, frame2, colors, labels, FLAGS)
            self.count += 1
        else:
            frame1, self.boxes1, self.confidences1, self.classids1, self.idxs1 = infer_image(net1, layer_names, \
                                height1, width1, frame1, colors, labels, FLAGS, self.boxes1, self.confidences1, self.classids1, self.idxs1, infer=False)
            frame2, self.boxes2, self.confidences2, self.classids2, self.idxs1 = infer_image(net2, layer_names, \
                                height2, width2, frame2, colors, labels, FLAGS, self.boxes2, self.confidences2, self.classids2, self.idxs2, infer=False)
            self.count = (self.count + 1) % 10

        # Flip horizontall and convert image to texture
        buf1 = cv2.flip(frame1, 0).tostring()
        img_texture1 = Texture.create(size=(frame1.shape[1], frame1.shape[0]), colorfmt='bgr')
        img_texture1.blit_buffer(buf1, colorfmt='bgr', bufferfmt='ubyte')
        buf2 = cv2.flip(frame2, 0).tostring()
        img_texture2 = Texture.create(size=(frame2.shape[1], frame2.shape[0]), colorfmt='bgr')
        img_texture2.blit_buffer(buf2, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam1.texture = img_texture1
        self.web_cam2.texture = img_texture2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model-path',
        type=str,
        default='./yolov3-coco/',
        help='The directory where the model weights and \
            configuration files are.')

    parser.add_argument('-w', '--weights',
        type=str,
        default='./yolov3-coco/yolov3.weights',
        help='Path to the file which contains the weights \
                for YOLOv3.')

    parser.add_argument('-cfg', '--config',
        type=str,
        default='./yolov3-coco/yolov3.cfg',
        help='Path to the configuration file for the YOLOv3 model.')

    parser.add_argument('-i', '--image-path',
        type=str,
        help='The path to the image file')

    parser.add_argument('-v', '--video-path',
        type=str,
        help='The path to the video file')


    parser.add_argument('-vo', '--video-output-path',
        type=str,
        default='./output.avi',
        help='The path of the output video file')

    parser.add_argument('-l', '--labels',
        type=str,
        default='./yolov3-coco/coco-labels',
        help='Path to the file having the \
                    labels in a new-line seperated way.')

    parser.add_argument('-c', '--confidence',
        type=float,
        default=0.5,
        help='The model will reject boundaries which has a \
                probabiity less than the confidence value. \
                default: 0.5')

    parser.add_argument('-th', '--threshold',
        type=float,
        default=0.3,
        help='The threshold to use when applying the \
                Non-Max Suppresion')

    parser.add_argument('--download-model',
        type=bool,
        default=False,
        help='Set to True, if the model weights and configurations \
                are not present on your local machine.')

    parser.add_argument('-t', '--show-time',
        type=bool,
        default=False,
        help='Show the time taken to infer each image.')

    FLAGS, unparsed = parser.parse_known_args()

    # Download the YOLOv3 models if needed
    if FLAGS.download_model:
        subprocess.call(['./yolov3-coco/get_model.sh'])

    # Get the labels
    labels = open(FLAGS.labels).read().strip().split('\n')

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configutation to form the pretrained YOLOv3 model
    # net = cv2.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)
    net1 = cv2.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)
    net2 = cv2.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

    # Get the output layer names of the model
    # layer_names = net.getLayerNames()
    # print( net.getUnconnectedOutLayers())
    # layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    layer_names = net1.getLayerNames()
    print( net1.getUnconnectedOutLayers())
    layer_names = [layer_names[i - 1] for i in net1.getUnconnectedOutLayers()]
    layer_names = net2.getLayerNames()
    print( net2.getUnconnectedOutLayers())
    layer_names = [layer_names[i - 1] for i in net2.getUnconnectedOutLayers()]
    
    CamApp().run()