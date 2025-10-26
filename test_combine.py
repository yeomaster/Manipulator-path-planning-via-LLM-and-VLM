import cv2
import torch
import socket
import numpy as np
import pyrealsense2 as rs
from transformers import CLIPProcessor, CLIPModel
from math import sqrt
import time

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


# Initialize Camera Intel RealSense
class DepthCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        profile = self.pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        depth_stream = profile.get_stream(rs.stream.depth)
        self.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False, None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return True, depth_image, color_image

    def get_3d_coordinates(self, x, y, depth_value):
        depth_in_meters = depth_value * self.depth_scale
        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth_in_meters)
        return point[0], point[1], point[2]


# Initialize the depth camera
dc = DepthCamera()


# Function to get CLIP features from the user prompt
def get_clip_features(prompt):
    inputs = clip_processor(text=prompt, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
    return features


# Function to create a TCP channel for the robotic arm
def create_tcp_channel(host, port, message):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            print(f"Connected to {host}:{port}")
            sock.sendall(message.encode('utf-8'))
            print("Command sent:", message)
            response = sock.recv(1024)
            print("Received:", response.decode())
    except ConnectionRefusedError:
        print(f"Connection to {host}:{port} refused.")
    except socket.error as e:
        print("Socket error:", e)


# Function to get averaged 3D coordinates
def get_average_3d_coordinates(x, y, depth_frame, num_samples=5):
    points = []
    for _ in range(num_samples):
        depth_value = depth_frame[y, x]
        point = dc.get_3d_coordinates(x, y, depth_value)
        points.append(point)

    avg_point = np.mean(points, axis=0)
    return avg_point[0], avg_point[1], avg_point[2]


# Function to calculate Euclidean distance in 3D space
def calculate_distance(x1, y1, z1, x2, y2, z2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


# Function to move the robot incrementally towards the target position
def move_robot_in_steps(target_x, target_y, target_z, step_size, host, port, dog_x=0, dog_y=0, dog_z=0):
    current_x, current_y, current_z = dog_x, dog_y, dog_z  # Start at the dog's position (reference point)

    while True:
        # Calculate the remaining distance between the current robot position and the target position
        remaining_distance = calculate_distance(current_x, current_y, current_z, target_x, target_y, target_z)
        print(f"Remaining distance to target: {remaining_distance:.2f} mm")

        # If the remaining distance is less than or equal to 200mm, stop moving
        if remaining_distance <= 150:
            print("Within 200mm of the target. No further movement required.")
            break

        # Calculate the direction vector to the target
        direction_x = target_x - current_x
        direction_y = target_y - current_y
        direction_z = target_z - current_z

        # Normalize the direction vector to get unit direction in each axis
        total_distance = sqrt(direction_x ** 2 + direction_y ** 2 + direction_z ** 2)
        unit_x = direction_x / total_distance
        unit_y = direction_y / total_distance
        unit_z = direction_z / total_distance

        # Calculate the next step position
        next_x = current_x + unit_x * step_size
        next_y = current_y + unit_y * step_size
        next_z = current_z + unit_z * step_size

        # Ensure that we don't overshoot the target
        if calculate_distance(dog_x, dog_y, dog_z, next_x, next_y, next_z) > remaining_distance:
            next_x, next_y, next_z = target_x, target_y, target_z

        # Send the movement command to the robot
        print(f"Moving to: X: {next_x:.2f}, Y: {next_y:.2f}, Z: {next_z:.2f}")
        message = (
            f"WayPoint,0,{next_x:.2f},{next_y:.2f},{next_z:.2f},180,00,180,0,0,90,0,90,0,TCP,Base,50,360,0,0,0,0,0,0,0,;"
        )
        create_tcp_channel(host, port, message)

        # Update the current position
        current_x, current_y, current_z = next_x, next_y, next_z

        # Wait for a short period before sending the next increment (to simulate real-time movement)
        time.sleep(1)  # Adjust time as needed


# Main loop
def main():
    user_prompt = input("Enter a prompt for CLIP: ")
    prompt_features = get_clip_features(user_prompt)

    cv2.namedWindow("Yolo frame")
    cv2.namedWindow("CLIP frame")

    dog_x, dog_y, dog_z = None, None, None
    prompted_x, prompted_y, prompted_z = None, None, None  # Coordinates for the prompted object
    command_sent = False  # Flag to ensure only one command is sent

    # Define the robot's host and port (replace with your robot's IP and port)
    target_host = "192.168.0.10"  # Example IP address
    target_port = 10003  # Example port

    while True:
        ret, depth_frame, Yolo_frame = dc.get_frame()
        if not ret:
            print("Failed to grab frame")
            break

        results = yolo_model(Yolo_frame)
        CLIP_frame = Yolo_frame.copy()

        for *box, conf, cls in results.xyxy[0].tolist():
            label = yolo_model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)

            # Calculate the center of the bounding box
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            # Get 3D coordinates of the object
            X, Y, Z = [int(coord * 1000) for coord in get_average_3d_coordinates(x_center, y_center, depth_frame)]

            # Detect the 'dog' and store its coordinates
            if label.lower() == "dog":
                dog_x, dog_y, dog_z = X, Y, Z

            # Get CLIP features for the detected object label
            label_features = get_clip_features(label)
            similarity = torch.cosine_similarity(prompt_features, label_features)

            # If similarity is high enough and the label matches the user prompt
            if similarity.item() > 0.5 and label.lower() in user_prompt.lower():
                prompted_x, prompted_y, prompted_z = X, Y, Z  # Capture the coordinates of the prompted object

                # If both the dog and prompted object are detected
                if dog_x is not None and prompted_x is not None:
                    # Calculate the relative position of the prompted object with respect to the dog
                    relative_x = prompted_x - dog_x
                    relative_y = prompted_y - dog_y
                    relative_z = prompted_z - dog_z

                    # Move the robot incrementally towards the prompted object
                    step_size = 10  # Step size in mm
                    move_robot_in_steps(relative_x, relative_y, relative_z, step_size, target_host, target_port)

                    command_sent = True  # Ensure the movement is only initiated once

            # Draw bounding box and label on the frame
            cv2.rectangle(CLIP_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(CLIP_frame, f"{label}: {conf:.2f}, X: {X}, Y: {Y}, Z: {Z}mm", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display frames
        cv2.imshow("Yolo frame", Yolo_frame)
        cv2.imshow("CLIP frame", CLIP_frame)

        # Exit loop when the 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()