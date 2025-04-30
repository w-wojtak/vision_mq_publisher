#!/usr/bin/env python3
import pika
import json
import time
from datetime import datetime
from enum import Enum, auto


class GraspState(Enum):
    IDLE = auto()
    GRASPING = auto()


class SimulatedVisionPublisher:
    def __init__(self):
        # RabbitMQ setup
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='hand_data')

        # Objects to detect and grasp
        self.objects = [
            {'id': 1, 'name': 'left_object', 'position': -40},
            {'id': 2, 'name': 'center_object', 'position': 0},
            {'id': 3, 'name': 'right_object', 'position': 40}
        ]

        # Track which objects have been grasped by each agent
        self.agent1_grasped_objects = set()
        self.agent2_grasped_objects = set()

        # Simulation parameters
        self.fps = 30
        self.frame_time = 1.0 / self.fps
        self.last_print_time = time.time()

        # State management for both agents
        self.agent1_state = GraspState.IDLE
        self.agent2_state = GraspState.IDLE
        self.agent1_grasped_object = None
        self.agent2_grasped_object = None

        # Timing parameters (in seconds)
        self.grasp_duration = 8.0     # Each grasp lasts 8 seconds
        self.idle_duration = 12.0     # 12 seconds between grasps
        self.agent_offset = 10.0      # 10 seconds offset between agents

        # Keep track of individual agent timings
        self.agent1_last_action = time.time()
        self.agent2_last_action = time.time() + self.agent_offset

    def get_next_available_object(self, agent_grasped_objects):
        """Get the next object that hasn't been grasped by this agent yet"""
        for obj in self.objects:
            if obj['id'] not in agent_grasped_objects:
                agent_grasped_objects.add(obj['id'])
                return obj
        return None

    def update_states(self):
        current_time = time.time()

        # Update Agent 1
        time_since_last_agent1 = current_time - self.agent1_last_action
        if self.agent1_state == GraspState.IDLE:
            if time_since_last_agent1 >= self.idle_duration:
                next_object = self.get_next_available_object(
                    self.agent1_grasped_objects)
                if next_object:
                    self.agent1_state = GraspState.GRASPING
                    self.agent1_grasped_object = next_object
                    self.agent1_last_action = current_time
        else:  # GRASPING
            if time_since_last_agent1 >= self.grasp_duration:
                self.agent1_state = GraspState.IDLE
                self.agent1_grasped_object = None
                self.agent1_last_action = current_time

        # Update Agent 2
        time_since_last_agent2 = current_time - self.agent2_last_action
        if self.agent2_state == GraspState.IDLE:
            if time_since_last_agent2 >= self.idle_duration:
                next_object = self.get_next_available_object(
                    self.agent2_grasped_objects)
                if next_object:
                    self.agent2_state = GraspState.GRASPING
                    self.agent2_grasped_object = next_object
                    self.agent2_last_action = current_time
        else:  # GRASPING
            if time_since_last_agent2 >= self.grasp_duration:
                self.agent2_state = GraspState.IDLE
                self.agent2_grasped_object = None
                self.agent2_last_action = current_time

    def generate_vision_data(self):
        self.update_states()

        data = {
            'timestamp': datetime.now().isoformat(),
            'frame_number': int(time.time() * self.fps),
            'agent1_state': self.agent1_state.name,
            'agent2_state': self.agent2_state.name,
            'agent1_grasped_object': (
                {'id': self.agent1_grasped_object['id'],
                 'name': self.agent1_grasped_object['name'],
                 'position': self.agent1_grasped_object['position']}
                if self.agent1_grasped_object else None
            ),
            'agent2_grasped_object': (
                {'id': self.agent2_grasped_object['id'],
                 'name': self.agent2_grasped_object['name'],
                 'position': self.agent2_grasped_object['position']}
                if self.agent2_grasped_object else None
            )
        }

        return data

    def run(self):
        print("Starting simulated vision publisher...")
        print(f"Grasp duration: {self.grasp_duration}s")
        print(f"Idle duration: {self.idle_duration}s")
        print(f"Offset between agents: {self.agent_offset}s")
        print("Each object will be grasped once by each agent")
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

                # Print status once per second
                current_time = time.time()
                if current_time - self.last_print_time >= 1.0:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
                    print(f"Agent 1 State: {data['agent1_state']}")
                    if data['agent1_grasped_object']:
                        print(
                            f"Agent 1 grasping: {data['agent1_grasped_object']['name']}")
                    print(
                        f"Objects grasped by Agent 1: {len(self.agent1_grasped_objects)}/3")

                    print(f"Agent 2 State: {data['agent2_state']}")
                    if data['agent2_grasped_object']:
                        print(
                            f"Agent 2 grasping: {data['agent2_grasped_object']['name']}")
                    print(
                        f"Objects grasped by Agent 2: {len(self.agent2_grasped_objects)}/3")
                    self.last_print_time = current_time

                # Check if all objects have been grasped by both agents
                if len(self.agent1_grasped_objects) == 3 and \
                   len(self.agent2_grasped_objects) == 3 and \
                   self.agent1_state == GraspState.IDLE and \
                   self.agent2_state == GraspState.IDLE:
                    print(
                        "\nAll objects have been grasped by both agents. Stopping simulation.")
                    break

                # Maintain frame rate
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
