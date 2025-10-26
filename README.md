**Manipulator Path planning via LLM and VLM**

**Explanation:**
- This project is an attempt to control the movement of 6-DOF Robotic manipulator using Visual Language models(VLM) and Large Language Models(LLM).
-	It is the combination of a Large Language Model (LLM) and a Vision Language Model (VLM) to create an adaptive program that can accept prompts from the user and direct the manipulator accordingly
-	For example: if we type in 'move dog to cup' or just 'cup', the manipulator will move the dog image as close as possible to the cup via path planning
-	Collision avoidance, i.e. making sure the robotic arm does not collide with another object on the WAY to the target, is the next code/step I plan to create/take

**Goals/Objectives**
- Understand and implement LLM and VLM
- Use a visual Language Model called YOLOv5 to detect objects seen from a camera
- Use a Large language Model called CLIP (from OpenAI) to match text prompts to objects in visuals.
- Understand and control the robotics manipulator

**Equipment used**
- Intel Realsense Camera – for xyz coordinate detection of objects
- Hansrobotv5 - 6DOF robotic manipulator

**Libraries used**
- YOLOv5 – used to detect objects in video in real-time 
- CLIP Model – Match text prompt to detect objects from YOLOv5 
- Pyrealsense2 - Library to connect intelrealsense camera to python program
<img width="757" height="568" alt="image" src="https://github.com/user-attachments/assets/8cc6ef8c-e4b7-45d3-834c-2d537ba1ef30" />


**How it works:**
<img width="1756" height="748" alt="image" src="https://github.com/user-attachments/assets/6bc0230c-8fa3-4c8d-b8e9-a11b4118b48b" />
-	The program asks for the user for a prompt, this prompt must contain an object for the program to focus on
-	YOLOv5 (VLM) will detect objects in the camera and create bounding boxes around them
- The prompt is interpreted by CLIP (LLM) and find’s matching objects in the YOLOv5 video feed

- The ‘dog’ will be used as a reference point (x,y,z = 0) and will be attached to the robotic arm
- ‘dog’ will be moved to the location of the prompted object
- So if I was to say: “move the dog to the mouse” the LLM will focus on 2 objects, a dog and a mouse. Meanwhile the VLM will create bounding boxes around all objects it can detect/identify in the video feed. Then only the bounding boxes with the identification of ‘dog’ or ‘mouse’ will be focused on. And finally, the robotic arm, which is the reference point, will start moving to the mouse.

<img width="731" height="566" alt="image" src="https://github.com/user-attachments/assets/1e445a23-87af-4e83-9cd8-04682da745e4" />


