
import pika
import json
import time
from datetime import datetime


def main():
    # Connect to RabbitMQ
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    # Create a queue
    queue_name = 'test_queue'
    channel.queue_declare(queue=queue_name)

    counter = 0
    try:
        while True:
            # Create a test message
            message = {
                'counter': counter,
                'timestamp': str(datetime.now()),
                'test_data': f'Hello from vision process! Message #{counter}'
            }

            # Convert to JSON and send
            json_message = json.dumps(message)
            channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=json_message
            )

            print(f" [x] Sent: {message}")

            counter += 1
            time.sleep(1)  # Wait 1 second between messages

    except KeyboardInterrupt:
        print("\nStopping publisher...")
    finally:
        connection.close()
        print("Connection closed")


if __name__ == "__main__":
    main()
