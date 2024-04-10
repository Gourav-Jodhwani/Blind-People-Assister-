from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from yolo_utils import infer_image

import cv2
import tensorflow as tf
import os
import numpy as np
import subprocess
import argparse

class StartPage(Screen):
    def __init__(self, **kwargs):
        super(StartPage, self).__init__(**kwargs)

        layout = BoxLayout(orientation='vertical')
        button_1 = Button(text="Start App with IP camera (Phone)", on_press=self.switch_to_phone_page)
        button_2 = Button(text="Start App with ESP camera", on_press=self.switch_to_esp_page)
        layout.add_widget(button_1)
        layout.add_widget(button_2)
        
        self.add_widget(layout)

    def switch_to_phone_page(self, *args):
        self.manager.current = 'phone_page'
    def switch_to_esp_page(self, *args):
        self.manager.current = 'esp_page'

class PhonePage(Screen):
    def __init__(self, **kwargs):
        super(PhonePage, self).__init__(**kwargs)

        layout = BoxLayout(orientation='vertical')

        button1 = Button(text='Show Live Video (Low FPS)', on_press=self.switch_to_phone_camera_vid_page)
        button2 = Button(text='High Inference Rate', on_press=self.switch_to_phone_camera_page)

        layout.add_widget(button1)
        layout.add_widget(button2)

        self.add_widget(layout)

    def switch_to_phone_camera_vid_page(self, *args):
        self.manager.current = 'phone_camera_vid_page'

    def switch_to_phone_camera_page(self, *args):
        self.manager.current = 'phone_camera_page'

class PhoneCameraVidPage(Screen):
    def __init__(self, **kwargs):
        super(PhoneCameraVidPage, self).__init__(**kwargs)

        self.web_cam = Image(size_hint=(1, .8))
        self.url_input = TextInput(hint_text="Enter URL here", size_hint=(1, 0.1))
        self.submit_button = Button(text="Submit", on_press=self.submit_url, size_hint=(1, 0.1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.url_input)
        layout.add_widget(self.submit_button)

        self.add_widget(layout)

        self.url = None
        self.capture = cv2.VideoCapture(self.url)
        self.count = 0
        self.boxes = None
        self.confidences = None
        self.classids = None
        self.idxs = None
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def submit_url(self, *args):
        self.url = self.url_input.text

    def update(self, *args):
        if self.url is None:
            return
        self.capture = cv2.VideoCapture(self.url)
        if self.capture is None or not self.capture.isOpened():
            return

        _, frame = self.capture.read()
        height, width = frame.shape[:2]

        if self.count == 0:
            frame, self.boxes, self.confidences, self.classids, self.idxs = infer_image(net, layer_names, height, width, frame, colors, labels,FLAGS)
        else:
            frame, self.boxes, self.confidences, self.classids, self.idxs = infer_image(net, layer_names,  height, width, frame, colors, labels,FLAGS, self.boxes, self.confidences, self.classids,self.idxs, infer=False)
            self.count = (self.count + 1) % 3

        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

class PhoneCameraPage(Screen):
    def __init__(self, **kwargs):
        super(PhoneCameraPage, self).__init__(**kwargs)

        self.web_cam = Image(size_hint=(1, .8))
        self.url_input = TextInput(hint_text="Enter URL here", size_hint=(1, 0.1))
        self.submit_button = Button(text="Submit", on_press=self.submit_url, size_hint=(1, 0.1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.url_input)
        layout.add_widget(self.submit_button)

        self.add_widget(layout)

        self.url = None
        self.capture = cv2.VideoCapture(self.url)
        self.count = 0
        self.boxes = None
        self.confidences = None
        self.classids = None
        self.idxs = None
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def submit_url(self, *args):
        self.url = self.url_input.text

    def update(self, *args):
        if self.url is None:
            return
        self.capture = cv2.VideoCapture(self.url)
        if self.capture is None or not self.capture.isOpened():
            return

        _, frame = self.capture.read()
        height, width = frame.shape[:2]

        if self.count == 0:
            frame, self.boxes, self.confidences, self.classids, self.idxs = infer_image(net, layer_names, height, width, frame, colors, labels,FLAGS)
        else:
            frame, self.boxes, self.confidences, self.classids, self.idxs = infer_image(net, layer_names,  height, width, frame, colors, labels,FLAGS, self.boxes, self.confidences, self.classids,self.idxs, infer=False)
            self.count = (self.count + 1) % 3

class ESPPage(Screen):
    def __init__(self, **kwargs):
        super(ESPPage, self).__init__(**kwargs)

        layout = BoxLayout(orientation='vertical')

        button1 = Button(text='Show Live Video (Low FPS)', on_press=self.switch_to_phone_camera_vid_page)
        button2 = Button(text='High Inference Rate', on_press=self.switch_to_phone_camera_page)

        layout.add_widget(button1)
        layout.add_widget(button2)

        self.add_widget(layout)

    def switch_to_phone_camera_vid_page(self, *args):
        self.manager.current = 'phone_camera_vid_page'

    def switch_to_phone_camera_page(self, *args):
        self.manager.current = 'phone_camera_page'

class MultiPageApp(App):
    def build(self):
        sm = ScreenManager()
        start_page = StartPage(name='start_page')
        # camera_page = CameraPage(name='camera_page')
        phone_page=PhonePage(name='phone_page')
        esp_page=ESPPage(name='esp_page')
        phone_camera_vid_page=PhoneCameraVidPage(name='phone_camera_vid_page')
        phone_camera_page=PhoneCameraPage(name='phone_camera_page')

        sm.add_widget(start_page)
        sm.add_widget(phone_page)
        sm.add_widget(esp_page)
        sm.add_widget(phone_camera_vid_page)
        sm.add_widget(phone_camera_page)
        # sm.add_widget(camera_page)

        return sm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model-path',
        type=str,
        default='./yolov3-coco/')

    parser.add_argument('-w', '--weights',
        type=str,
        default='./yolov3-coco/yolov3.weights')

    parser.add_argument('-cfg', '--config',
        type=str,
        default='./yolov3-coco/yolov3.cfg')

    parser.add_argument('-i', '--image-path',
        type=str)

    parser.add_argument('-v', '--video-path',
        type=str,)

    parser.add_argument('-vo', '--video-output-path',
        type=str,
        default='./output.avi')

    parser.add_argument('-l', '--labels',
        type=str,
        default='./yolov3-coco/coco-labels')

    parser.add_argument('-c', '--confidence',
        type=float,
        default=0.5)

    parser.add_argument('-th', '--threshold',
        type=float,
        default=0.3)

    parser.add_argument('--download-model',
        type=bool,
        default=False)

    parser.add_argument('-t', '--show-time',
        type=bool,
        default=False)

    FLAGS, unparsed = parser.parse_known_args()

    # Download the YOLOv3 models if needed
    if FLAGS.download_model:
        subprocess.call(['./yolov3-coco/get_model.sh'])

    # Get the labels
    labels = open(FLAGS.labels).read().strip().split('\n')

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv2.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    print( net.getUnconnectedOutLayers())
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    MultiPageApp().run()
