
import cv2
import pika
import os
import json
import utils.functions as functions
import multiprocessing
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import ast
import statistics

model_hand_path = os.getenv("MODEL_PATH_FILE") + "/" + os.getenv("GESTURE_NAME_FILE")
model_gesture_path = os.getenv("MODEL_PATH_FILE") + "/" + os.getenv("GESTURE_MODEL_FILE")

class_gesture = ast.literal_eval(os.getenv("GESTRUE_CLASS"))
conf_gesture  = float(os.getenv("CONF_GESTURE"))

def rabbitmq_listener(queue, stop_event, shared_data):
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(
                os.getenv("RMQ_HOST"),
                os.getenv("RMQ_PORT"),
                credentials = pika.PlainCredentials(os.getenv("RMQ_USER"), os.getenv("RMQ_PASS"))
            ))
        channel = connection.channel()
    except Exception as e:
        print(f"Error connected to RabbitMQ {e}")
        exit()
    channel.queue_declare(queue=queue, durable=True)
    def callback(ch, method, properties, body):
        try:
            message = body.decode('utf-8')
            print(f"Message received: {message}")
            data =  json.loads(body)
            shared_data["data"] = data
        except Exception as e:
            shared_data["data"] = ""
            print (f"Error in message received: {e}")

    channel.basic_consume(queue=queue, on_message_callback=callback, auto_ack=True)
    print(f"Waiting for messages in the queue '{queue}'...")
    while not stop_event.is_set():
        try:
            connection.process_data_events(time_limit=1)
        except pika.exceptions.AMQPConnectionError:
            print("RabbitMQ connection closed.")
            break
    connection.close()
    print("RabbitMQ process stopped.")
    

def publish_message(queue, message):
    try:
        message = json.dumps(message)
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(
                    os.getenv("RMQ_HOST"),
                    os.getenv("RMQ_PORT"),
                    credentials = pika.PlainCredentials(os.getenv("RMQ_USER"), os.getenv("RMQ_PASS"))
                ))
            channel = connection.channel()
        except Exception as e:
            print(f"Error connected to RabbitMQ {e}")
            exit()
        channel.queue_declare(queue=queue, durable=True)

        channel.basic_publish(
            exchange='',
            routing_key=queue,
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=2 
            )
        )
        print(f"Message Send: {message}")
    except Exception as e:
        print(f"Error at publish message: {e}")


def read_webcam_frames(stop_event, shared_data):

    base_options = python.BaseOptions(model_asset_path= model_hand_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    interpreter = functions.tf.lite.Interpreter(model_path=model_gesture_path)
    interpreter.allocate_tensors()

    cap = cv2.VideoCapture(0)#functions.string_connexion)
    if not cap.isOpened():
        print("Error")
        return
    
    gestures = []
    qtd_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error")
            break
        if True: # len(shared_data["data"]) > 0 and shared_data["data"] ['action'] == 'GESTURE':
            num_frames = 0# int(shared_data["data"] ['num_frames'])
            gesture_res = '?'
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_ = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            results = detector.detect(frame_)
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    landmark_coordinates = functions.extract_landmark_coordinates(frame, hand_landmarks)
                    normalized_landmarks = functions.normalize_landmark_coordinates(landmark_coordinates)
                    normalized_landmarks = np.array([normalized_landmarks]).flatten()  
                    normalized_landmarks = normalized_landmarks.astype(np.float32) 
                    gesture = functions.gesture_inference(interpreter, normalized_landmarks)
                    gestures.append(gesture)
                    #if gesture >= conf_gesture:
                    qtd_frames +=1
                    if qtd_frames > num_frames:
                        gesture_res = class_gesture[statistics.mode(gestures)]
                        #publish_message(os.getenv("RMQ_QUEUE_RESULT"), str(result))
                        shared_data["data"] = ""
                        qtd_frames = 0
                        gestures = []
            
            frame = functions.draw_landmarks(frame_.numpy_view(), results, gesture_res)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.namedWindow("Webcam Frame", cv2.WINDOW_NORMAL)
        cv2.imshow('Webcam Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    cap.release()
    cv2.destroyAllWindows()





if __name__=="__main__":
    
    manager = multiprocessing.Manager()
    stop_event = multiprocessing.Event()

    shared_data = manager.dict()
    shared_data["data"] = ""

    p1 = multiprocessing.Process(target=rabbitmq_listener, args=(os.getenv("RMQ_QUEUE_GESTURE"), stop_event, shared_data, ))
    p2 = multiprocessing.Process(target=read_webcam_frames, args=( stop_event, shared_data))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
