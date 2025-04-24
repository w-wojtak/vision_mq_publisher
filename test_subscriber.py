
import pika
import json


def callback(ch, method, properties, body):
    # Decode and print received message
    message = json.loads(body)
    print(f" [x] Received: {message}")


def main():
    # Connect to RabbitMQ
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    # Create the same queue as publisher
    queue_name = 'test_queue'
    channel.queue_declare(queue=queue_name)

    # Set up subscription
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.basic_consume(
        queue=queue_name,
        auto_ack=True,
        on_message_callback=callback
    )

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\nStopping subscriber...")
    finally:
        connection.close()
        print("Connection closed")


if __name__ == "__main__":
    main()
