import pika
import json
import time
import random
from datetime import datetime
from enum import Enum, auto


class GraspState(Enum):
    SCANNING = auto()        # Looking for objects
    APPROACHING = auto()     # Moving towards selected object
    GRASPING = auto()        # Attempting to grasp
    HOLDING = auto()         # Successfully grasped
    RELEASING = auto()       # Releasing object


class SimulatedVisionPublisher:
    def __init__(self):
        # RabbitMQ setup
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='hand_data')

        # Objects to detect and grasp
        self.objects = [
            {'id': 1, 'name': 'cup', 'color': 'blue', 'size': 'medium'},
            {'id': 2, 'name': 'ball', 'color': 'red', 'size': 'small'},
            {'id': 3, 'name': 'box', 'color': 'green', 'size': 'large'}
        ]

        # Simulation parameters
        self.fps = 30
        self.frame_time = 1.0 / self.fps
        self.processing_time = 0.03

        # State management
        self.current_state = GraspState.SCANNING
        self.detected_objects = []
        self.selected_object = None
        self.grasp_progress = 0.0  # 0.0 to 1.0
        self.state_start_time = time.time()

        # Position simulation (simplified 2D)
        self.hand_position = {'x': 0.0, 'y': 0.0}
        self.object_positions = self._generate_object_positions()

    def _generate_object_positions(self):
        # Generate random positions for objects (simplified 2D)
        positions = {}
        for obj in self.objects:
            positions[obj['id']] = {
                'x': random.uniform(-1.0, 1.0),
                'y': random.uniform(-1.0, 1.0)
            }
        return positions

    def _update_state(self):
        current_time = time.time()
        elapsed = current_time - self.state_start_time

        if self.current_state == GraspState.SCANNING:
            # Simulate object detection
            self.detected_objects = []
            for obj in self.objects:
                if random.random() < 0.95:  # 95% detection rate
                    obj_data = obj.copy()
                    obj_data['position'] = self.object_positions[obj['id']]
                    obj_data['confidence'] = random.uniform(0.85, 0.98)
                    self.detected_objects.append(obj_data)

            if self.detected_objects and elapsed > 2.0:
                self.selected_object = random.choice(self.detected_objects)
                self.current_state = GraspState.APPROACHING
                self.state_start_time = current_time

        elif self.current_state == GraspState.APPROACHING:
            # Simulate approaching the object
            target_pos = self.selected_object['position']
            dx = target_pos['x'] - self.hand_position['x']
            dy = target_pos['y'] - self.hand_position['y']

            # Update hand position
            speed = 0.1
            self.hand_position['x'] += dx * speed
            self.hand_position['y'] += dy * speed

            # Check if we're close enough
            distance = (dx**2 + dy**2)**0.5
            if distance < 0.1 or elapsed > 3.0:
                self.current_state = GraspState.GRASPING
                self.state_start_time = current_time
                self.grasp_progress = 0.0

        elif self.current_state == GraspState.GRASPING:
            # Simulate grasping motion
            self.grasp_progress = min(1.0, elapsed / 1.5)

            if self.grasp_progress >= 1.0:
                self.current_state = GraspState.HOLDING
                self.state_start_time = current_time

        elif self.current_state == GraspState.HOLDING:
            if elapsed > 2.0:
                self.current_state = GraspState.RELEASING
                self.state_start_time = current_time
                self.grasp_progress = 1.0

        elif self.current_state == GraspState.RELEASING:
            # Simulate release motion
            self.grasp_progress = max(0.0, 1.0 - elapsed / 1.0)

            if self.grasp_progress <= 0.0:
                self.current_state = GraspState.SCANNING
                self.state_start_time = current_time
                self.selected_object = None
                # Reset hand position
                self.hand_position = {'x': 0.0, 'y': 0.0}

    def generate_vision_data(self):
        self._update_state()

        data = {
            'timestamp': datetime.now().isoformat(),
            'frame_number': int(time.time() * self.fps),
            'state': self.current_state.name,
            'hand_position': self.hand_position,
            'detected_objects': self.detected_objects,
            'processing_time_ms': round(self.processing_time * 1000, 2)
        }

        if self.selected_object:
            data['selected_object'] = self.selected_object
            data['grasp_progress'] = round(self.grasp_progress, 3)

        return data

    def run(self):
        print("Starting simulated vision publisher...")
        print("Press Ctrl+C to stop")

        try:
            while True:
                start_time = time.time()

                # Generate and send data
                data = self.generate_vision_data()
                self.channel.basic_publish(
                    exchange='',
                    routing_key='hand_data',
                    body=json.dumps(data)
                )
                print(f" [x] State: {data['state']}")
                if 'selected_object' in data:
                    print(f"     Object: {data['selected_object']['name']}")
                    print(
                        f"     Grasp Progress: {data.get('grasp_progress', 0):.2f}")

                # Simulate processing time
                time.sleep(self.processing_time)

                # Wait for next frame
                elapsed = time.time() - start_time
                if elapsed < self.frame_time:
                    time.sleep(self.frame_time - elapsed)

        except KeyboardInterrupt:
            print("\nStopping publisher...")
        finally:
            self.connection.close()
            print("Connection closed")


if __name__ == "__main__":
    publisher = SimulatedVisionPublisher()
    publisher.run()
