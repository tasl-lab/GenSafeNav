import os
import gym
import numpy as np
from numpy.linalg import norm
import copy
import csv
import glob
import imageio
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import patches

import random

from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs import *
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.state import JointState

class CrowdSimVarNum(CrowdSim):
    """
    The environment for our model with no trajectory prediction, or the baseline models with no prediction
    The number of humans at each timestep can change within a range
    """
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super().__init__()
        self.id_counter = None
        self.observed_human_ids = None
        self.pred_method = None

        self.all_possible_aggr_factor = [-3, -2, -1, 0, 1, 2, 3]


    def configure(self, config):
        """ read the config to the environment variables """
        super(CrowdSimVarNum, self).configure(config)
        self.action_type=config.action_space.kinematics


    # set observation space and action space
    def set_robot(self, robot):
        self.robot = robot

        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need

        d={}
        # robot node: px, py, r, gx, gy, v_pref, theta
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,7,), dtype = np.float32)
        # only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_human_num, 2), dtype=np.float32)
        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)
        # whether each human is visible to robot (ordered by human ID, should not be sorted)
        d['visible_masks'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.max_human_num,),
                                            dtype=np.bool_)
        
        d['aggressiveness_factor'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(d)

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)


    # set robot initial state and generate all humans for reset function
    # for crowd nav: human_num == self.human_num
    # for leader follower: human_num = self.human_num - 1
    def generate_robot_humans(self, phase, human_num=None):
        if self.record:
            px, py = 0, 0
            gx, gy = 0, -1.5
            self.robot.set(px, py, gx, gy, 0, 0, np.pi / 2)
            # generate a dummy human
            for i in range(self.max_human_num):
                human = Human(self.config, 'humans')
                human.set(15, 15, 15, 15, 0, 0, 0)
                human.isObstacle = True
                self.humans.append(human)

        else:
            # for sim2real
            if self.robot.kinematics == 'unicycle':
                # generate robot
                angle = np.random.uniform(0, np.pi * 2)
                px = self.arena_size * np.cos(angle)
                py = self.arena_size * np.sin(angle)
                while True:
                    gx, gy = np.random.uniform(-self.arena_size, self.arena_size, 2)
                    if np.linalg.norm([px - gx, py - gy]) >= 4:  # 1 was 6
                        break
                self.robot.set(px, py, gx, gy, 0, 0, np.random.uniform(0, 2 * np.pi))  # randomize init orientation
                # 1 to 4 humans
                self.human_num = np.random.randint(1, self.config.sim.human_num + self.human_num_range + 1)
                # print('human_num:', self.human_num)
                # self.human_num = 4


            # for sim exp
            else:
                # generate robot
                while True:
                    px, py, gx, gy = np.random.uniform(-self.arena_size, self.arena_size, 4)
                    if np.linalg.norm([px - gx, py - gy]) >= 8: # 6
                        break
                self.robot.set(px, py, gx, gy, 0, 0, np.pi / 2)
                if self.aggressiveness_factor is None:
                    aggr_index = np.random.randint(len(self.all_possible_aggr_factor))
                    aggr_factor = self.all_possible_aggr_factor[aggr_index]
                    self.robot.set_aggr_factor(aggr_factor)
                    self.robot.set_aggr_index(aggr_index)
                else:
                    self.robot.set_aggr_factor(self.aggressiveness_factor)
                    self.robot.set_aggr_index(self.all_possible_aggr_factor.index(self.aggressiveness_factor))

                # generate humans
                self.human_num = np.random.randint(low=self.config.sim.human_num - self.human_num_range,
                                                   high=self.config.sim.human_num + self.human_num_range + 1)


            self.generate_random_human_position(human_num=self.human_num)
            self.last_human_states = np.zeros((self.human_num, 5))
            # set human ids
            for i in range(self.human_num):
                self.humans[i].id = i



    # generate a human that starts on a circle, and its goal is on the opposite side of the circle
    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()

        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            noise_range = 2
            px_noise = np.random.uniform(0, 1) * noise_range
            py_noise = np.random.uniform(0, 1) * noise_range
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False

            for i, agent in enumerate([self.robot] + self.humans):
                # keep human at least 3 meters away from robot
                if self.robot.kinematics == 'unicycle' and i == 0:
                    min_dist = self.circle_radius / 2  # Todo: if circle_radius <= 4, it will get stuck here
                else:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break

        human.set(px, py, -px, -py, 0, 0, 0)

        return human

    # calculate the ground truth future trajectory of humans
    # if robot is visible: assume linear motion for robot
    # ret val: [self.predict_steps + 1, self.human_num, 4]
    # method: 'truth' or 'const_vel' or 'inferred'
    def calc_human_future_traj(self, method):
        # if the robot is invisible, it won't affect human motions
        # else it will
        human_num = self.human_num + 1 if self.robot.visible else self.human_num
        # buffer to store predicted future traj of all humans [px, py, vx, vy]
        # [time, human id, features]
        if method == 'truth':
            self.human_future_traj = np.zeros((self.buffer_len + 1, human_num, 4))
        elif method == 'const_vel':
            self.human_future_traj = np.zeros((self.predict_steps + 1, human_num, 4))
        else:
            raise NotImplementedError

        # initialize the 0-th position with current states
        for i in range(self.human_num):
            # use true states for now, to count for invisible humans' influence on visible humans
            # take px, py, vx, vy, remove radius
            self.human_future_traj[0, i] = np.array(self.humans[i].get_observable_state_list()[:-1])

        # if we are using constant velocity model, we need to use displacement to approximate velocity (pos_t - pos_t-1)
        # we shouldn't use true velocity for fair comparison with GST inferred pred
        if method == 'const_vel':
            self.human_future_traj[0, :, 2:4] = self.prev_human_pos[:, 2:4]

        # add robot to the end of the array
        if self.robot.visible:
            self.human_future_traj[0, -1] = np.array(self.robot.get_observable_state_list()[:-1])

        if method == 'truth':
            for i in range(1, self.buffer_len + 1):
                for j in range(self.human_num):
                    # prepare joint state for all humans
                    full_state = np.concatenate(
                        (self.human_future_traj[i - 1, j], self.humans[j].get_full_state_list()[4:]))
                    observable_states = []
                    for k in range(self.human_num):
                        if j == k:
                            continue
                        observable_states.append(
                            np.concatenate((self.human_future_traj[i - 1, k], [self.humans[k].radius])))

                    # use joint states to get actions from the states in the last step (i-1)
                    action = self.humans[j].act_joint_state(JointState(full_state, observable_states))

                    # step all humans with action
                    self.human_future_traj[i, j] = self.humans[j].one_step_lookahead(
                        self.human_future_traj[i - 1, j, :2], action)

                if self.robot.visible:
                    action = ActionXY(*self.human_future_traj[i - 1, -1, 2:])
                    # update px, py, vx, vy
                    self.human_future_traj[i, -1] = self.robot.one_step_lookahead(self.human_future_traj[i - 1, -1, :2],
                                                                                  action)
            # only take predictions every self.pred_interval steps
            self.human_future_traj = self.human_future_traj[::self.pred_interval]
        # for const vel model
        elif method == 'const_vel':
            # [self.pred_steps+1, human_num, 4]
            self.human_future_traj = np.tile(self.human_future_traj[0].reshape(1, human_num, 4), (self.predict_steps+1, 1, 1))
            # [self.pred_steps+1, human_num, 2]
            pred_timestep = np.tile(np.arange(0, self.predict_steps+1, dtype=float).reshape((self.predict_steps+1, 1, 1)) * self.time_step * self.pred_interval,
                                    [1, human_num, 2])
            pred_disp = pred_timestep * self.human_future_traj[:, :, 2:]
            self.human_future_traj[:, :, :2] = self.human_future_traj[:, :, :2] + pred_disp
        else:
            raise NotImplementedError

        # remove the robot if it is visible
        if self.robot.visible:
            self.human_future_traj = self.human_future_traj[:, :-1]


        # remove invisible humans
        self.human_future_traj[:, np.logical_not(self.human_visibility), :2] = 15
        self.human_future_traj[:, np.logical_not(self.human_visibility), 2:] = 0

        return self.human_future_traj


    # reset = True: reset calls this function; reset = False: step calls this function
    # sorted: sort all humans by distance to robot or not
    def generate_ob(self, reset, sort=False):
        """Generate observation for reset and step functions"""
        ob = {}

        # nodes
        visible_humans, num_visibles, self.human_visibility = self.get_num_human_in_fov()
        ob['robot_node'] = self.robot.get_full_state_list_noV()

        prev_human_pos = copy.deepcopy(self.last_human_states)
        self.update_last_human_states(self.human_visibility, reset=reset)

        # edges
        ob['temporal_edges'] = np.array([self.robot.vx, self.robot.vy])

        # ([relative px, relative py, disp_x, disp_y], human id)
        all_spatial_edges = np.ones((self.max_human_num, 2)) * np.inf

        for i in range(self.human_num):
            if self.human_visibility[i]:
                # vector pointing from human i to robot
                relative_pos = np.array(
                    [self.last_human_states[i, 0] - self.robot.px, self.last_human_states[i, 1] - self.robot.py])
                all_spatial_edges[self.humans[i].id, :2] = relative_pos

        ob['visible_masks'] = np.zeros(self.max_human_num, dtype=np.bool_)
        # sort all humans by distance (invisible humans will be in the end automatically)
        if sort:
            ob['spatial_edges'] = np.array(sorted(all_spatial_edges, key=lambda x: np.linalg.norm(x)))
            # after sorting, the visible humans must be in the front
            if num_visibles > 0:
                ob['visible_masks'][:num_visibles] = True
        else:
            ob['spatial_edges'] = all_spatial_edges
            ob['visible_masks'][:self.human_num] = self.human_visibility
        ob['spatial_edges'][np.isinf(ob['spatial_edges'])] = 15
        ob['detected_human_num'] = num_visibles
        if ob['detected_human_num'] == 0:
            ob['detected_human_num'] = 1

        self.observed_human_ids = np.where(self.human_visibility)[0]

        ob['aggressiveness_factor'] = np.array([[self.robot.get_aggr_factor()]])

        self.ob = ob

        return ob



    # Update the specified human's end goals in the environment randomly
    def update_human_pos_goal(self, human):
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            v_pref = 1.0 if human.v_pref == 0 else human.v_pref
            gx_noise = (np.random.random() - 0.5) * v_pref
            gy_noise = (np.random.random() - 0.5) * v_pref
            gx = self.circle_radius * np.cos(angle) + gx_noise
            gy = self.circle_radius * np.sin(angle) + gy_noise
            collide = False

            if not collide:
                break

        # Give human new goal
        human.gx = gx
        human.gy = gy


    def reset(self, phase='train', test_case=None):
        """
        Reset the environment
        :return:
        """
        for h in self.humans:
            h.reset_aci(self.config.aci_related.alpha)
        
        # original version conformal prediction
        self.robot.past_predictions.clear()
        self.robot.past_locations.clear()
        for h in self.humans:
            h.past_predictions.clear()
            h.past_locations.clear()
        
        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case=self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case # test case is passed in to calculate specific seed to generate case
        self.global_time = 0
        self.step_counter = 0
        self.id_counter = 0


        self.humans = []

        self.observed_human_ids = []

        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        # here we use a counter to calculate seed. The seed=counter_offset + case_counter
        self.rand_seed = counter_offset[phase] + self.case_counter[phase] + self.thisSeed
        np.random.seed(self.rand_seed)

        self.generate_robot_humans(phase)

        # record px, py, r of each human, used for crowd_sim_pc env
        self.cur_human_states = np.zeros((self.max_human_num, 3))
        for i in range(self.human_num):
            self.cur_human_states[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius])

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]

        # initialize potential and angular potential
        rob_goal_vec = np.array([self.robot.gx, self.robot.gy]) - np.array([self.robot.px, self.robot.py])
        self.potential = -abs(np.linalg.norm(rob_goal_vec))
        self.angle = np.arctan2(rob_goal_vec[1], rob_goal_vec[0]) - self.robot.theta
        if self.angle > np.pi:
            # self.abs_angle = np.pi * 2 - self.abs_angle
            self.angle = self.angle - 2 * np.pi
        elif self.angle < -np.pi:
            self.angle = self.angle + 2 * np.pi

        # get robot observation
        ob = self.generate_ob(reset=True, sort=self.config.args.sort_humans)

        return ob


    def step(self, action, update=True):
        raise NotImplementedError

    def calc_reward(self, action, danger_zone='circle'): 
        dmin = float('inf')
        danger_dists = []
        
        collision = False

        # collision check with humans
        for i, human in enumerate(self.humans):
            dx = human.px - self.robot.px
            dy = human.py - self.robot.py
            closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - human.radius - self.robot.radius

            if closest_dist < self.discomfort_dist:
                danger_dists.append(closest_dist)
                
                
            if closest_dist < 0:
                collision = True             
                break
            
            elif closest_dist < dmin:
                dmin = closest_dist


        # check if reaching the goal
        if self.robot.kinematics == 'unicycle':
            goal_radius = 0.6
        else:
            goal_radius = self.robot.radius
        reaching_goal = norm(
            np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position())) < goal_radius

        # use danger_zone to determine the condition for Danger
        if danger_zone == 'circle' or self.phase == 'train':
            danger_cond = dmin < self.discomfort_dist
            min_danger_dist = 0
        else:
            # if the robot collides with future states, give it a collision penalty
            relative_pos = self.human_future_traj[1:, :, :2] - np.array([self.robot.px, self.robot.py])
            relative_dist = np.linalg.norm(relative_pos, axis=-1)

            collision_idx = relative_dist < self.robot.radius + self.config.humans.radius  # [predict_steps, human_num]

            danger_cond = np.any(collision_idx)
            # if robot is dangerously close to any human, calculate the min distance between robot and its closest human
            if danger_cond:
                min_danger_dist = np.amin(relative_dist[collision_idx])
            else:
                min_danger_dist = 0

        if self.global_time >= self.time_limit - 1:
            reward = 0
            cost = 0
            done = True
            episode_info = Timeout()
        elif collision:
            reward = self.config.reward.base_collision_penalty 
            cost = 0 # self.config.reward.base_collision_penalty 
            done = True
            episode_info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            cost = 0
            done = True
            episode_info = ReachGoal()

        elif danger_cond:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            # print(dmin)
            reward = 0 #(dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step # tag: 05/05/2024
            cost = 0 # (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step # 0 #tag: 05/05/2024 #today
            done = False
            episode_info = Danger(min_danger_dist)

        else:
            # potential reward
            if self.robot.kinematics == 'holonomic':
                pot_factor = 2
            else:
                pot_factor = 3
            potential_cur = np.linalg.norm(
                np.array([self.robot.px, self.robot.py]) - np.array(self.robot.get_goal_position()))
            reward = pot_factor * (-abs(potential_cur) - self.potential) # when moving towards the goal, the reward is positve
            cost = 0
            
            self.potential = -abs(potential_cur)

            done = False
            episode_info = Nothing()

        if self.robot.kinematics == 'unicycle':
            raise NotImplementedError

        cost = 0 
        cost = -cost
        return reward, done, [episode_info, cost]


    def render(self, mode='human'):
        raise NotImplementedError
    # def animate_episode(self, save_path, filename="default_name"):
    #     if not os.path.exists(save_path):
    #         print(f"path: {save_path} is empty!!! Skip animate the pics")

    #     all_pic_filenames = os.path.join(save_path, "*.png")
    #     generated_gif_filename = os.path.join(save_path, f"{filename}.gif")

    #     filenames = glob.glob(all_pic_filenames)
    #     filenames.sort()

    #     images = []
    #     for filename in filenames:
    #         images.append(imageio.imread(filename))
    #         os.remove(filename)

    #     imageio.mimsave(generated_gif_filename, images)
    def animate_episode(self, save_path, filename="default_name"):
        if not os.path.exists(save_path):
            print(f"path: {save_path} is empty!!! Skip animate the pics")
            return

        all_pic_filenames = os.path.join(save_path, "*.png")
        generated_mp4_filename = os.path.join(save_path, f"{filename}.mp4")

        filenames = glob.glob(all_pic_filenames)
        filenames.sort()

        images = []
        for filename in filenames:
            image = imageio.imread(filename)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            elif image.shape[2] != 3:
                raise ValueError(f"Unexpected image shape: {image.shape}")

            height, width = image.shape[:2]
            new_height = (height + 15) // 16 * 16
            new_width = (width + 15) // 16 * 16
            image = np.pad(image, ((0, new_height - height), (0, new_width - width), (0, 0)), mode='constant')

            images.append(image)
            os.remove(filename)

        writer = imageio.get_writer(generated_mp4_filename, fps=10, codec='libx264')
        for image in images:
            writer.append_data(image)
        writer.close()


    def plot_step(self, save_path):
        self.plot_scenario(-1, save_path) 

    def plot_scenario(self, alert_agent_id, save_path):
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = "#ffd079"
        goal_color = '#f56200'
        arrow_color_robot = "#ff6600"
        arrow_color_human = '#073f93'
        human_color = '#96cbfd'
        buffer_color = '#bee7fa'
        buffer_alpha_near = 0.6
        buffer_alpha_away = 0.2
        range_color = '#afaeae'
        text_color = '#7d7d7d'
        arrow_style = patches.ArrowStyle("simple", head_length=4, head_width=2)

        def calcFOVLineEndPoint(ang, point, extendFactor):
            FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                [np.sin(ang), np.cos(ang), 0],
                                [0, 0, 1]])
            point.extend([1])
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint

        fig, ax = plt.subplots(figsize=(7, 7), dpi=300)  # Increased dpi for higher resolution
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xticks([])  # Hide x-axis numbers
        ax.set_yticks([])  # Hide y-axis numbers
        artists = []

        sensor_range = plt.Circle(self.robot.get_position(), self.robot.sensor_range + self.robot.radius + self.config.humans.radius, fill=False, color=range_color, linestyle='--')
        ax.add_artist(sensor_range)
        artists.append(sensor_range)

        goal = mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal', alpha=1.0)
        ax.add_artist(goal)
        artists.append(goal)

        for human in self.humans:
            if not self.config.aci_related.only_prediction_line:
                buffer_circle = plt.Circle(human.get_position(), human.radius + self.config.aci_related.current_position_buffer, fill=True, linewidth=1.5, color=buffer_color, alpha=buffer_alpha_near)
                ax.add_artist(buffer_circle)
                artists.append(buffer_circle)
            if human.last_prediction is None: 
                continue
            if self.config.aci_related.only_circular:
                continue
            for i, point in enumerate(human.last_prediction[1:]):
                buffer_alpha = buffer_alpha_near if i < 2 else buffer_alpha_away
                if not self.config.aci_related.only_prediction_line:
                    circle = plt.Circle(point, human.last_aci_predicted_conformity_score[i] + human.radius, fill=True, linewidth=1.5, color=buffer_color, alpha=buffer_alpha)
                else:
                    circle = plt.Circle(point, human.radius, fill=True, linewidth=1.5, color=buffer_color, alpha=buffer_alpha_away*1.5)
                ax.add_artist(circle)
                artists.append(circle)

        robotX, robotY = self.robot.get_position()
        robot = plt.Circle((robotX, robotY), self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot)
        artists.append(robot)

        radius = self.robot.radius

        if self.robot.FOV < 2 * np.pi:
            FOVAng = self.robot_fov / 2
            FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
            FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')

            startPointX = robotX
            startPointY = robotY
            endPointX = robotX + radius * np.cos(robot_theta)
            endPointY = robotY + radius * np.sin(robot_theta)

            FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
            FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

            ax.add_artist(FOVLine1)
            ax.add_artist(FOVLine2)
            artists.append(FOVLine1)
            artists.append(FOVLine2)

        human_circles = [plt.Circle(human.get_position(), human.radius, fill=True) for human in self.humans]
        actual_arena_size = self.arena_size + 0.5

        for i in range(len(self.humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])
            if self.human_visibility[i]:
                human_circles[i].set_color(c=human_color)
            else:
                human_circles[i].set_color(c=human_color)

        arrowStartEnd = []
        robot_theta = self.robot.theta if self.robot.kinematics == 'unicycle' else np.arctan2(self.robot.vy, self.robot.vx)
        arrowStartEnd.append(((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

        for i, human in enumerate(self.humans):
            theta = np.arctan2(human.vy, human.vx)
            arrowStartEnd.append(((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

        arrows = []
        for i, arrow in enumerate(arrowStartEnd):
            arrow_color = arrow_color_human if i > 0 else arrow_color_robot
            arrows.append(patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style))
        for arrow in arrows:
            ax.add_artist(arrow)
            artists.append(arrow)

        plt.savefig(os.path.join(save_path, f"{datetime.now()}.png"), bbox_inches='tight')
        plt.close()

        for item in artists:
            item.remove()
        for t in ax.texts:
            t.set_visible(False)
    
    
    def append_value_to_csv(self, value, column_name, filename):
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([column_name])
                
            writer.writerow([value])
            
    