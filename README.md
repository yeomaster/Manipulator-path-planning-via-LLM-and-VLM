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
YOLOv5 – used to detect objects in video in real-time 
CLIP Model – Match text prompt to detect objects from YOLOv5 

<img width="757" height="568" alt="image" src="https://github.com/user-attachments/assets/8cc6ef8c-e4b7-45d3-834c-2d537ba1ef30" />
