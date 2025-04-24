
import cv2
import pika
import os
import json
import utils.functions as functions
import multiprocessing
import mediapipe as mp
import numpy as np
import ast

from tensorflow.keras.models import load_model

model_hand_path = os.getenv("MODEL_PATH_FILE") + "/" + os.getenv("GESTURE_NAME_FILE")
model_gesture_path = os.getenv("MODEL_PATH_FILE") + "/" + os.getenv("GESTURE_MODEL_FILE_LTMS")

class_gesture = ast.literal_eval(os.getenv("GESTRUE_CLASS"))

conf_gesture  = float(os.getenv("CONF_GESTURE"))

seq_length = 30

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

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    model = load_model(model_gesture_path)

    cap = cv2.VideoCapture(functions.string_connexion)
    if not cap.isOpened():
        print("Error")
        return
    
    seq = []
    action_seq = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error")
            break
        if True: #len(shared_data["data"]) > 0 and shared_data["data"] ['action'] == 'GESTURE':
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    joint = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in hand_landmarks.landmark])
                    idx_v1 = np.array([0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19])
                    idx_v2 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
                    v = joint[idx_v2, :3] - joint[idx_v1, :3]
                    norms = np.linalg.norm(v, axis=1, keepdims=True)
                    v = np.divide(v, norms, where=(norms != 0)) 
                    idx_angle_1 = np.array([0,1,2,4,5,6,8,9,10,12,13,14,16,17,18])
                    idx_angle_2 = np.array([1,2,3,5,6,7,9,10,11,13,14,15,17,18,19])
                    angle = np.degrees(np.arccos(np.einsum('nt,nt->n', v[idx_angle_1], v[idx_angle_2])))
                    d = np.concatenate([joint.flatten(), angle])
                    seq.append(d)

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    if len(seq) < seq_length:
                        continue
                    input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                    y_pred = model.predict(input_data).squeeze()
                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]
                    if conf < 0.5:
                        continue
                    action = class_gesture[i_pred]
                    action_seq.append(action)

                    if len(action_seq) < 3:
                        continue

                    this_action = ''
                    if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                        this_action = action
                    cv2.putText(frame, f'{this_action.upper()}', org=(int(hand_landmarks.landmark[0].x * frame.shape[1]), int(hand_landmarks.landmark[0].y * frame.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                    

                    '''
                    qtd_frames +=1
                    if qtd_frames > num_frames:
                        result = class_gesture[statistics.mode(gestures)]
                        publish_message(os.getenv("RMQ_QUEUE_RESULT"), str(result))
                        shared_data["data"] = ""
                        qtd_frames = 0
                        gestures = []
                        cv2.putText(frame, str(result), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                    '''

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
