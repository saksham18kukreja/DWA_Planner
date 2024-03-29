# Dynamic Window Approach Planner

This is the code for a Dynamic Window Approach Planner based on kinematic bicycle model.

The code takes input of the pose of the vehicle, the pose of immediate goals and samples a set of kinematic bicycle trajectories.

It then generates a window around the vehicle and avoids the obstacle only when present inside this window.

The optimal steering angle of the vehicle to avoid these obstacles and follow the desired path is dependent on the final cost value taking in account 3 different costs

## 1. Heading Error/cost 
The heading error of the vehicle is calculated as the difference of the final yaw of the vehicle at the end of theses sampled trajectories and the pose of the goal point.
The cost for each individual trajectories is calculated by the formula 

vector to the end point of trajectory : $[traj_n - pose_v]$, where $n \in \text{1:trajectory length}$ and $pose_v$ is the current vehicle pose

vector to the goal  : $[pose_g - pose_v]$ where $pose_g$ is the pose of the goal point and $pose_v$ is the current vehicle pose

heading angle is the dot product between $vector_g$ and $vector_t$

heading cost = $(1+\text{heading angle})/2$ to include the cost between 0 and 1

## 2. Obstacle Cost
Obstacle cost is simply the euclidean distance between the final point of the trajctory with each obstacle inside the dynamic window.
Highest cost is given the obstacle with the least euclidean distance

Obstacle cost: $1-(1/\text{total distance})$

## 3. Desired Speed Cost
Desired Speed cost is the used set the target velocity for the vehicle and is given by $\text{vehicle current velocity}/ \text{desired velocity}$

Total cost for each trajectory is based on a weighted sum of each individual cost and weights are set based on the requirements.

The trajectory with the highest cost is selected and the steering angle of that trajectory is used to direct the vehicle.

## Media
### 1. Trajectory following and obstacle avoidance
![video](https://github.com/saksham18kukreja/DWA_Planner/blob/main/media/video.gif)

The planner also works on a receding horizon approach to follow the next goal point(red circles) when the previous goal has been reached<br>
The obstacles are in blue color.<br>
"1.5" on the top is the desired velocity in m/sec

### 2. Dynamic Window
![video](https://github.com/saksham18kukreja/DWA_Planner/blob/main/media/video_window.gif)

The dynamic window is generated based on the current vehicle position and the size of the window is an adjustable parameter.
