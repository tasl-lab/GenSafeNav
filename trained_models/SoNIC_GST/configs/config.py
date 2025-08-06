import numpy as np


class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):
    # general configs for OpenAI gym env
    env = BaseConfig()
    reward = BaseConfig()
    aci_related = BaseConfig()
    constrained_rl_related = BaseConfig()
    network_related = BaseConfig() #TODO: whether or not to incorporate aci scores/addbias/etc
    dataset = BaseConfig()
    policy = BaseConfig()
    #################### frequently tuned parameters ######################## 
    aci_related.considered_steps = 2
    aci_related.prediction_extra_buffer_size = 0.0
    aci_related.current_position_buffer = 0.25
    aci_related.alpha = 0.1

    aci_related.noise_clip_for_conformity_scores = 0#0.1
    aci_related.noise_std_for_conformity_scores = 0#0.1
    
    aci_related.noise_std_for_cost = 0.0
    aci_related.noise_clip_for_cost = 0.00
    aci_related.only_circular = False #tag
    aci_related.only_prediction_line = False
    constrained_rl_related.cost_limit = 0.4
    constrained_rl_related.lag_init = 0.10#0.125
    constrained_rl_related.lag_lr = 16e-4#18e-4
    
    policy.aci_input = True
    policy.constant_std = True
    
    policy.constrain_cost = True
    
    env.val_size = 100
    env.test_size = 500
    note = f"last_cp_0_08_cl_{constrained_rl_related.cost_limit}_lag_{constrained_rl_related.lag_lr}_lag_init_{constrained_rl_related.lag_init}"
    #################### unfrequently tuned ######################## 
    aggressiveness_factor = 0.0 # unused for now
    reward.intrusion_start_dist = 0.50 # unused


    env.randomize_attributes = True
    env.time_limit = 50  # yjp mark: original: 50
    env.time_step = 0.25    
    # record robot states and actions an episode
    #       for system identification in sim2real
    env.record = False
    env.load_act = False

    # config for reward function
    
    reward.success_reward = 10
    reward.base_collision_penalty = -20
    # discomfort distance
    reward.discomfort_dist = 0.25
    # today
    reward.discomfort_penalty_factor = 10
    reward.gamma = 0.99    
    
    # for now, import all args from arguments.py
    cv_prediction_steps = 5 # unused to be removed

    training = BaseConfig()
    training.device = "cuda:0"

    # config for simulation
    sim = BaseConfig()
    sim.circle_radius = 6 * np.sqrt(2)
    sim.arena_size = 6
    sim.human_num = 20
    # actual human num in each timestep, in [human_num-human_num_range, human_num+human_num_range]
    sim.human_num_range = 0
    sim.predict_steps = 5  # yjp mark: prediction horizon (steps)
    # 'const_vel': constant velocity model,
    # 'truth': ground truth future traj (with info in robot's fov)
    # 'inferred': inferred future traj from GST network
    # 'none': no prediction
    sim.predict_method = 'inferred'
    # render the simulation during training or not
    sim.render = False

    # for save_traj only
    render_traj = False
    save_slides = False
    save_path = None

    # whether wrap the vec env with VecPretextNormalize class
    # = True only if we are using a network for human trajectory prediction (sim.predict_method = 'inferred')
    if sim.predict_method == 'inferred':
        env.use_wrapper = True
    else:
        env.use_wrapper = False

    # human config
    humans = BaseConfig()
    humans.visible = True
    # orca or social_force for now
    humans.policy = "orca"
    humans.radius = 0.3
    humans.v_pref = 1
    humans.sensor = "coordinates"
    # FOV = this values * PI
    humans.FOV = 2.

    # a human may change its goal before it reaches its old goal
    # if randomize human behaviors, set to True, else set to False
    humans.random_goal_changing = True
    humans.goal_change_chance = 0.5

    # a human may change its goal after it reaches its old goal
    humans.end_goal_changing = True
    humans.end_goal_change_chance = 1.0

    # a human may change its radius and/or v_pref after it reaches its current goal
    humans.random_radii = False
    humans.random_v_pref = False

    # one human may have a random chance to be blind to other agents at every time step
    humans.random_unobservability = False
    humans.unobservable_chance = 0.3

    humans.random_policy_changing = False

    # robot config
    robot = BaseConfig()
    # whether robot is visible to humans (whether humans respond to the robot's motion)
    robot.visible = False # tag: 05/02/2024
    # For baseline: srnn; our method: selfAttn_merge_srnn
    robot.policy = 'selfAttn_merge_srnn'
    robot.radius = 0.3
    robot.v_pref = 1
    robot.sensor = "coordinates"
    # FOV = this values * PI
    robot.FOV = 2
    # radius of perception range
    robot.sensor_range = 5

    # action space of the robot
    action_space = BaseConfig()
    # holonomic or unicycle
    action_space.kinematics = "holonomic"

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.15
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    # config for social force
    sf = BaseConfig()
    sf.A = 2.
    sf.B = 1
    sf.KI = 1

    # config for data collection for training the GST predictor
    data = BaseConfig()
    data.tot_steps = 40000
    data.render = False
    data.collect_train_data = False
    data.num_processes = 5
    data.data_save_dir = 'gst_updated/datasets/orca_20humans_no_rand'
    # number of seconds between each position in traj pred model
    data.pred_timestep = 0.25

    # config for the GST predictor
    pred = BaseConfig()
    # see 'gst_updated/results/README.md' for how to set this variable
    # If randomized humans: gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj
    # else: gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000/sj
    pred.model_dir = 'gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj'

    # LIDAR config
    lidar = BaseConfig()
    # angular resolution (offset angle between neighboring rays) in degrees
    lidar.angular_res = 5
    # range in meters
    lidar.range = 10

    # config for sim2real
    sim2real = BaseConfig()
    # use dummy robot and human states or not
    sim2real.use_dummy_detect = True
    sim2real.record = False
    sim2real.load_act = False
    sim2real.ROSStepInterval = 0.03
    sim2real.fixed_time_interval = 0.1
    sim2real.use_fixed_time_interval = True

    if sim.predict_method == 'inferred' and env.use_wrapper == False:
        raise ValueError("If using inferred prediction, you must wrap the envs!")
    if sim.predict_method != 'inferred' and env.use_wrapper:
        raise ValueError("If not using inferred prediction, you must NOT wrap the envs!")
