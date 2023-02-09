## **AteltaSDK**

![alt text](assets/banner.png)



Open source is what we love the most. Existing mediapipe only helps us to detect and draw the pose. But just detecting poses and 
drawing it is not necessarily the most important and implemented on every problem. Sometimes, we need to do a lot on top of it. Some classic 
examples include:

- Counting the number of times a specific pose is detected. Examples include push ups and pull ups counter.
- Pose angle and distance calculations
- Pose matching 
- Speed calculation etc...

So in order to make stuff more flexible and more customizable, we introduce AteltaSDK. A free open source wrapper of mediapipe. Our main 
goal is to provide extensibility for already existing features provided by mediapipe and build more on the top of it. 

Fun fact: Even our Atelta's core API services also use this sdk for all the backend computations. 