import numpy as np
import torch

from crowd_sim.envs.utils.info import *


def run_with_visualization(actor_critic, eval_envs, num_processes, device, test_size, config, args, gif_save_path):
    """ function to run all testing episodes and log the testing metrics """
    # initializations
    eval_episode_rewards = []

    if config.robot.policy not in ['orca', 'social_force']:
        eval_recurrent_hidden_states = {}

        node_num = 1
        edge_num = actor_critic.base.human_num + 1
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, actor_critic.base.human_node_rnn_size,
                                                                     device=device)

        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                           actor_critic.base.human_human_edge_rnn_size,
                                                                           device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    # success_times = []
    # collision_times = []
    # timeout_times = []
    #
    # success = 0
    # collision = 0
    # timeout = 0
    # too_close_ratios = []
    # min_dist = []
    #
    # collision_cases = []
    # timeout_cases = []
    #
    # all_path_len = []

    # to make it work with the virtualenv in sim2real
    if hasattr(eval_envs.venv, 'envs'):
        baseEnv = eval_envs.venv.envs[0].env
    else:
        baseEnv = eval_envs.venv.unwrapped.envs[0].env
    time_limit = baseEnv.time_limit

    # start the testing episodes
    for k in range(test_size):
        baseEnv.episode_k = k
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0
        obs = eval_envs.reset()
        out_pred = obs['spatial_edges'][:, :, :].to('cpu').numpy()[0]
        outs = baseEnv.talk2Env(out_pred)
        aci_predicted_conformity_scores, aci_cost = outs#np.array([o[0] for o in outs]) # [num_envs, num_humans, num_pred_steps]
        aci_predicted_conformity_scores = np.array([aci_predicted_conformity_scores])
        obs['conformity_scores'] = torch.from_numpy(aci_predicted_conformity_scores).to(torch.float32).to(device)
        global_time = 0.0
        path_len = 0.
        too_close = 0.
        last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()


        while not done:
            stepCounter = stepCounter + 1
            if config.robot.policy not in ['orca', 'social_force']:
                # run inference on the NN policy
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)
            else:
                action = torch.zeros([1, 2], device=device)
            if not done:
                global_time = baseEnv.global_time

            baseEnv.plot_step(gif_save_path) 

            # Obser reward and next obs
            obs, rew, done, infos = eval_envs.step(action)

            out_pred = obs['spatial_edges'][:, :, 2:].to('cpu').numpy()
            # send manager action to all processes
            out_pred = obs['spatial_edges'][:, :, :].to('cpu').numpy()[0]
            outs = baseEnv.talk2Env(out_pred)
            aci_predicted_conformity_scores, aci_cost = outs#np.array([o[0] for o in outs]) # [num_envs, num_humans, num_pred_steps]
            aci_predicted_conformity_scores = np.array([aci_predicted_conformity_scores])
            obs['conformity_scores'] = torch.from_numpy(aci_predicted_conformity_scores).to(torch.float32).to(device)
            # render

            # record the info for calculating testing metrics
            rewards.append(rew)

            path_len = path_len + np.linalg.norm(obs['robot_node'][0, 0, :2].cpu().numpy() - last_pos)
            last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()

            episode_rew += rew[0]


            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        # an episode ends!
        print('')
        print('Reward={}'.format(episode_rew))
        print('Episode', k, 'ends in', stepCounter)

        if isinstance(infos[0]['info'], ReachGoal):
            # success += 1
            # success_times.append(global_time)
            print('Success')
            result = "success"
        elif isinstance(infos[0]['info'], Collision):
            # collision += 1
            # collision_cases.append(k)
            # collision_times.append(global_time)
            print('Collision')
            result = "collision"
        elif isinstance(infos[0]['info'], Timeout):
            # timeout += 1
            # timeout_cases.append(k)
            # timeout_times.append(time_limit)
            print('Time out')
            result = 'time_out'
        elif isinstance(infos[0]['info'] is None):
            result = 'none'
            pass
        else:
            raise ValueError('Invalid end signal from environment')

        baseEnv.animate_episode(gif_save_path, f"{k}_{result}")

    eval_envs.close()