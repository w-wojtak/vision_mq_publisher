import pika
import json
import time
from datetime import datetime


def main():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    queue_name = 'test_queue'
    channel.queue_declare(queue=queue_name)

    counter = 0
    try:
        while True:
            message = {
                'counter': counter,
                'timestamp': str(datetime.now()),
                'test_data': f'Hello from vision_mq_publisher! Message #{counter}'
            }
            channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=json.dumps(message)
            )
            print(f" [x] Sent: {message}")
            counter += 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping publisher...")
    finally:
        connection.close()
        print("Connection closed")


if __name__ == "__main__":
    main()
