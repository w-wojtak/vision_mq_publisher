import cv2
import numpy as np
import os
from dotenv import load_dotenv
import pandas as pd
from ultralytics import YOLO
import os
import torch
import ast
from threading import Lock
import pika
import json


from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import copy
import itertools

import time
import tensorflow as tf


lock = Lock()

load_dotenv()


string_connexion = int(os.getenv("STRING_CONNECTION"))

action = os.getenv("ACTION")
filename = os.getenv("FILENAME")
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)



connection    = channel = None
exchange      = os.getenv("RMQ_EXCHANGE")
exchange_type = os.getenv("RMQ_EXCHANGETYPE")




def draw_landmarks(image, result, gesture_res):
  hand_landmarks_list = result.hand_landmarks
  handedness_list = result.handedness
  annotated_image = np.copy(image)
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN
    cv2.putText(annotated_image, f"{gesture_res}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


def extract_landmark_coordinates(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_points = []
    for landmark in landmarks:  
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_points.append([landmark_x, landmark_y])

    return landmark_points

def normalize_landmark_coordinates(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0]
    for index, (x, y) in enumerate(temp_landmark_list):
        temp_landmark_list[index] = [x - base_x, y - base_y]
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list), default=1) 
    return [n / max_value for n in temp_landmark_list]


def gesture_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], np.array([input_data]))
    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]['index'])
    #print(tflite_results)
    #print("Predict:", np.argmax(np.squeeze(tflite_results)))
    return np.argmax(np.squeeze(tflite_results))
