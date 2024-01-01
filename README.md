# Dynamic Window Approach Planner

This is the code for a Dynamic Window Approach Planner based on kinematic bicycle model.

The code takes input of the pose of the vehicle, the pose of immediate goals and samples a set of kinematic bicycle trajectories.

It then generates a window around the vehicle and avoids the obstacle only when present inside this window.

The optimal steering angle of the vehicle to avoid these obstacles and follow the desired path is dependent on the final cost value taking in account 3 different costs

# 1. Heading error 
The heading error of the vehicle is calculated as the difference of the final yaw of the vehicle at the end of theses sampled trajectories and the pose of the goal point.
The cost for each individual trajectories is calculated by the formula 

vector to the goal : $[traj_n - goal]$
