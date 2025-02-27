from math import radians
from sys import setrecursionlimit
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree
from operator import add

length_car =   2
breadth_car = 1
velocity_upper = 1.5 
velocity_lower = 0
pi = 3.14
MAX_STEERING_ANGLE = pi/6
steering_angle = np.arange(-MAX_STEERING_ANGLE,MAX_STEERING_ANGLE,0.02)
L = length_car
lr = 1.4
time = 2.5   
dt = 0.5    
robot_radius = 3
obstacle_radius = 1 
future_goal = 30

# vehicle = VehicleFootprint()
vehicle_boundary = []
velocity_resolution = 0.05

# DT = 1/20
MAX_ACC = 1
track_pt = -1

A = 1 #from global path
B = 1 #from goal point

dyn_x1 = 0
dyn_y1 = 0
velocity_obstacle1 = 0
dyn_x2 = 0
dyn_y2 = 0    
velocity_obstacle2 = 0
dyn_x3 = 0
dyn_y3 = 0
velocity_obstacle3 = 4
TARGET_VELOCITY = 0.8
GOAL_THRESHOLD = 1
num_digit = 15
# local_goal = 100

GOAL_COST_GAIN = 5
SPEED_COST_GAIN = 1 
OBS_COST_GAIN = 1

window_size = 9
thin_fact = 3


SIM_LOOP = 500
plot_trajectory = 1
show_window = False


class state:
    def __init__(self,x, y, yaw, velocity):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.velocity = velocity


def new_coordinate(wheel,theta):
    x_new = wheel[0]*np.cos(theta) - wheel[1]*np.sin(theta)
    y_new = wheel[0]*np.sin(theta) + wheel[1]*np.cos(theta)
    return [x_new,y_new]


def physical_window(current_state):
    min_x = current_state.x
    max_x = current_state.x + window_size
    min_y = current_state.y - window_size/thin_fact
    max_y = current_state.y + window_size/thin_fact
    return [min_x, max_x, min_y, max_y]


# the wheels are thought of as the outline of the vehicle
def generate_window(current_state):

    fl_wheel = [window_size,-(window_size/thin_fact)]
    fr_wheel = [window_size,(window_size/thin_fact)]
    rl_wheel = [0,-(window_size/thin_fact)]
    rr_wheel = [0,(window_size/thin_fact)]
    

    fl_wheel = new_coordinate(fl_wheel,current_state.yaw)
    fr_wheel = new_coordinate(fr_wheel,current_state.yaw)
    rl_wheel = new_coordinate(rl_wheel,current_state.yaw)
    rr_wheel = new_coordinate(rr_wheel,current_state.yaw)
    


    outline = np.array([fl_wheel,fr_wheel,rr_wheel,rl_wheel,fl_wheel])
    x_outline = outline[:,0] + current_state.x
    y_outline = outline[:,1] + current_state.y
    
    max_x = np.max(x_outline)
    min_x = np.min(x_outline)
    max_y = np.max(y_outline)
    min_y = np.min(y_outline)

    dim = [min_x, max_x, min_y, max_y]
    # print(dim)

    return x_outline,y_outline, dim

def append_obs_list(ob,box):
    obs_list = []
    for pt in ob:
        if (pt[0] > box[0] and pt[0] < box[1] and pt[1] > box[2] and pt[1] < box[3]):
            obs_list.append(pt)

    return obs_list


# function to generate the dynamic window 
def Window_gen(curr_velocity):
    min_velocity = np.max([curr_velocity - MAX_ACC*dt, velocity_lower])
    max_velocity = np.min([curr_velocity + MAX_ACC*dt, velocity_upper])

    return min_velocity, max_velocity


def plot_car(current_state):
    
    fl_wheel = [length_car/2,-(breadth_car/2)]
    fr_wheel = [length_car/2,(breadth_car/2)]
    rl_wheel = [-length_car/2,-(breadth_car/2)]
    rr_wheel = [-length_car/2,(breadth_car/2)]
    

    fl_wheel = new_coordinate(fl_wheel,current_state.yaw)
    fr_wheel = new_coordinate(fr_wheel,current_state.yaw)
    rl_wheel = new_coordinate(rl_wheel,current_state.yaw)
    rr_wheel = new_coordinate(rr_wheel,current_state.yaw)
    

    outline = np.array([fl_wheel,fr_wheel,rr_wheel,rl_wheel,fl_wheel])
    x_outline = outline[:,0] + current_state.x
    y_outline = outline[:,1] + current_state.y
  
    return x_outline,y_outline


def reference_path(x,y):
    f = CubicSpline(x,y,bc_type='natural')
    return f


def generate_trajectory(curr_state, velocity, steering):
    
    traj_pt = []
    for t in np.arange(0,time,dt):
        temp_pt = state(0,0,0,0)
        theta_dot = velocity*np.tan(steering)*t/L
        temp_pt.yaw = curr_state.yaw + theta_dot
        temp_pt.x = curr_state.x + velocity*np.cos(temp_pt.yaw)*t
        temp_pt.y = curr_state.y + velocity*np.sin(temp_pt.yaw)*t
        temp_pt.velocity = np.round(velocity,num_digit)
        traj_pt.append(temp_pt)

    return traj_pt


def calc_to_goal_cost(traj, goal, current_state):
    theta_goal = np.arctan(goal.y/goal.x)
    traj_vec = [traj[track_pt].x - current_state.x, traj[track_pt].y - current_state.y]
    goal_vec = [goal.x - current_state.x, goal.y - current_state.y]
    unit_traj_vec = traj_vec/np.linalg.norm(traj_vec)
    unit_goal_vec = goal_vec/np.linalg.norm(goal_vec)
    heading = np.dot(unit_traj_vec, unit_goal_vec)
    heading = (1 + heading)/2

    return round(heading,num_digit)


def calc_to_speed_cost(traj, velocity):
    # return abs(velocity - abs(traj[-1].velocity))
    return round(abs(traj[1].velocity)/velocity, num_digit)
 

def calc_to_obs_cost(traj,obs_list):

    if (not obs_list):
        cost = 1
        # print("no obstacle")
        return cost

    else:
        total_dist = 0
        for pt in obs_list:
            dist = np.sqrt((traj[-1].x - pt[0])**2 + (traj[-1].y - pt[1])**2)
            total_dist+=dist

        cost = 1 - (1/total_dist)
        return round(cost, num_digit)

    
def main_dwa_loop(goal, obs_list, current_state):
    min_cost = 1e6
    min_obs_cost = min_cost
    min_goal_cost = min_cost
    min_speed_cost = min_cost

    trajectory = []
    best_traj = []
    final_cost_list = []



    for steering in steering_angle:
        traj = []
        traj_point = state(current_state.x,current_state.y,current_state.yaw,0)
        

        traj = generate_trajectory(traj_point, velocity_upper, steering)
        trajectory.append(traj)

        goal_cost = calc_to_goal_cost(traj, goal, current_state)
        speed_cost = calc_to_speed_cost(traj, velocity_upper)
        obstacle_cost = calc_to_obs_cost(traj, obs_list)
        final_cost = GOAL_COST_GAIN*goal_cost + SPEED_COST_GAIN*speed_cost + OBS_COST_GAIN*obstacle_cost
        final_cost = round(1/(1+np.exp(-final_cost)),num_digit)
        final_cost_list.append(final_cost)

    # print(len(trajectory))
    best_ind = np.argmax(final_cost_list)
   
    return trajectory[best_ind], trajectory


def main_loop(current_state,goal, obs_list):

    next_state = state(1,0,0,0)
    # window_vel_min, window_vel_max = Window_gen(current_state.velocity)
    # win_dimension = physical_window(current_state)

    best_traj, all_traj = main_dwa_loop(goal, obs_list, current_state)

   
    goal_thresh = np.sqrt((current_state.x - goal.x)**2 + (current_state.y - goal.y)**2)
    # print(goal_thresh)

    if (goal_thresh > GOAL_THRESHOLD):
        next_state = best_traj[1]

    else:
        next_state.velocity = 0
        next_state.yaw = np.arctan(abs(current_state.y - goal.y/current_state.x - goal.x))     

    # print("current state values updated")
    return next_state, best_traj, all_traj


def generate_waypoints():
    lower_bound = 0
    upper_bound = 150
    num_points = 10
    points_x = np.linspace(lower_bound, upper_bound, num_points)
    points_y = np.random.uniform(-3, 3, num_points)
    
    # Sort the points to ensure they are strictly increasing
    points_x.sort()
    
    return points_x, points_y



def main():
   
    # wx = [-90.0, -70.5, -35.0, -20.5, -10.0, 0.0]
    # wx = [-5.0, 30.0, 40, 60, 80, 81, 82]
    wx, wy = generate_waypoints()
    # wy = [0.0, 0.0, 0.0, 0.0, 0.0, -5.0, -15.0]

    ob = np.array(([20.0, 3.0],[20.0, 0.0],[50.0,0.0]))
    # ob = np.array([[50.0, 0.0]])
   
        
    spline = reference_path(wx,wy)
    x_spline = np.arange(wx[0],wx[-1],0.1)
    y_spline = spline(x_spline)
    x_road_spline = x_spline
    y_road_spline_upper_bound = y_spline + 10
    y_road_spline_lower_bound = y_spline - 10


    global_path_data = [(x_spline[i],y_spline[i])for i in range(len(x_spline))]
    
    current_state = state(wx[0],wy[0],0,0)
  
    tree = KDTree(global_path_data)

    
    area = 15
    i=0
    print("starting")
    j=1

    current_state_buffer = []
    current_goal_buffer = []
    selected_traj_buffer = []
    
    for i in range(SIM_LOOP):
        # plt.cla()

        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        # goal = state(wx[j],wy[j],0,0)
        # goal = state(x_spline[j],y_spline[j],0,0)
        # print(goal.x, goal.y)

        # query the next goal from the tree 
        _, idx = tree.query((current_state.x, current_state.y))
        # print(idx)

        last_goal = len(x_spline)-1
        goal_track_pt = min(last_goal, idx + future_goal)
        goal = state(x_spline[goal_track_pt], y_spline[goal_track_pt],0,0)


        obs_x = ob[:,0]
        obs_y = ob[:,1]
       

        bd_x, bd_y, box_dimension = generate_window(current_state)
        obs_list = append_obs_list(ob,box_dimension)
        obs_list= []

        nt_state, select_traj, all_traj = main_loop(current_state, goal, obs_list)
        
        current_state_buffer.append(current_state)
        current_goal_buffer.append(goal)
        selected_traj_buffer.append(select_traj)


        current_state = nt_state
        x_car,y_car = plot_car(current_state)
        

        #checks the next point of the global points
        if np.sqrt((current_state.x - goal.x)**2 + (current_state.y - goal.y)**2) <= GOAL_THRESHOLD:
            if (goal_track_pt==len(x_spline)-1):
                return current_state_buffer, current_goal_buffer, selected_traj_buffer
                # break  
            # else:
            #     j+=1 


        # plotting functions 
        # select_traj_x = []
        # select_traj_y = []

        # if (plot_trajectory):
        #     black_traj_x = []
        #     black_traj_y = []

        #     # all trajectories
        #     for traj in all_traj:
        #         temp_traj_x = []
        #         temp_traj_y = []
        #         for pt in traj:
        #             temp_traj_x.append(pt.x)
        #             temp_traj_y.append(pt.y)
        #         black_traj_x.append(temp_traj_x)
        #         black_traj_y.append(temp_traj_y)

        #     plt.plot(np.array(black_traj_x),np.array(black_traj_y),'--',color='red')


        # # the selected trajectory 
        # for pt in select_traj:
        #     select_traj_x.append(pt.x)
        #     select_traj_y.append(pt.y)

        # if (show_window):
        #     plt.plot(bd_x,bd_y,'-',color='red')


        # plt.plot(np.array(select_traj_x),np.array(select_traj_y),'-',color='black')
        # plt.plot(obs_x,obs_y,'o',color='blue')
        # plt.plot(goal.x,goal.y,'o',color='red')
        # plt.plot(x_car,y_car,'-')        
        # plt.plot(x_spline,y_spline,'-')
        # plt.plot(x_road_spline,y_road_spline_upper_bound,'-',color='black')
        # plt.plot(x_road_spline,y_road_spline_lower_bound,'-',color='black')
        # plt.grid(True)
        # plt.xlim(nt_state.x - area, nt_state.x + area)
        # plt.ylim(nt_state.y - area, nt_state.y + area)
        # plt.title(current_state.velocity)
        # plt.pause(0.000000001)
        

    # print("finished")


if __name__ == "__main__":
    main()