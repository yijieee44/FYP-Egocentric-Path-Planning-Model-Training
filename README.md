# Egocentric Path Planning For Vehicle

Egocentric view is the view that human perceive and make decision. Path planning for vehicle is a process of predicting where the vehicle will be in the future. There are two types of planning which are lateral planning and longitudinal planning. Lateral planning is the local horizontal movement of the vehicle, which is for controlling the steering. Longitudinal planning is the local vertical movement of the vehicle which is for controlling the gas and brake command.

In this work, an end-to-end path planning model which considered both lateral and longitudinal planning by using a single highway image is trained. The input of the model is a highway image while the output of the model is a list of 100 3D coordinates.

Sample images with predicted path:

![alt text](https://github.com/yijieee44/FYP-Egocentric-Path-Planning-Model-Training/blob/main/resource/samples_day.png)
![alt text](https://github.com/yijieee44/FYP-Egocentric-Path-Planning-Model-Training/blob/main/resource/samples_night.png)


## Reference

[[End-to-end lateral planning](https://blog.comma.ai/end-to-end-lateral-planning/)]
