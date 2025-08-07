import os
import shutil
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import wandb

from rl import ppo
from rl.networks import network_utils
from arguments import get_args
from rl.networks.envs import make_vec_envs
from rl.networks.model import Policy
from rl.networks.storage_safe import RolloutStorage

from crowd_nav.configs.config import Config

def main():
    """
    Main function for training a robot policy network using PPO with cost constraints
    for crowd navigation environments.
    """
    # Read command line arguments and environment configuration
    algo_args = get_args()
    env_config = config = Config()
    
    # Create unique model name based on configuration parameters
    model_name = f"{env_config.note}_seed_{algo_args.seed}_curr_buffer_{env_config.aci_related.current_position_buffer}_c_l_{env_config.constrained_rl_related.cost_limit}_clip_param_{algo_args.clip_param}_considered_steps_{env_config.aci_related.considered_steps}_alpha_{env_config.aci_related.alpha}_noise_{env_config.aci_related.noise_clip_for_conformity_scores}_{env_config.aci_related.noise_clip_for_cost}"
    env_config.model_name = model_name
    algo_args.output_dir = f"trained_models/{model_name}"
    
    # Create directory structure for saving training outputs
    if not os.path.exists(algo_args.output_dir):
        os.makedirs(algo_args.output_dir)
    # Prevent accidental overwriting of existing models
    elif not algo_args.overwrite:
        raise ValueError('output_dir already exists!')

    # Save configuration files for reproducibility
    save_config_dir = os.path.join(algo_args.output_dir, 'configs')
    if not os.path.exists(save_config_dir):
        os.makedirs(save_config_dir)
    shutil.copy('crowd_nav/configs/config.py', save_config_dir)
    shutil.copy('crowd_nav/configs/__init__.py', save_config_dir)
    shutil.copy('arguments.py', algo_args.output_dir)

    # Set random seeds for reproducibility
    torch.manual_seed(algo_args.seed)
    torch.cuda.manual_seed_all(algo_args.seed)
    if algo_args.cuda:
        if algo_args.cuda_deterministic:
            # Reproducible but slower execution
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            # Not reproducible but faster execution
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    # Configure PyTorch threading and device
    torch.set_num_threads(algo_args.num_threads)
    device = torch.device("cuda" if algo_args.cuda else "cpu")
    env_name = algo_args.env_name

    # Special configuration for rendering mode
    if config.sim.render:
        algo_args.num_processes = 1
        algo_args.num_mini_batch = 1
    
    env_config.env.num_processes = algo_args.num_processes
    env_config.args = algo_args

    # Create vectorized environment for parallel training
    envs = make_vec_envs(env_name, algo_args.seed, algo_args.num_processes,
                         algo_args.gamma, None, device, False,
                         config=env_config, ax=None,
                         pretext_wrapper=config.env.use_wrapper)

    # Create main policy network (actor-critic)
    actor_critic = Policy(
        envs.observation_space.spaces,  # Dict observation space
        envs.action_space,
        env_config,
        base_kwargs=algo_args,
        base=config.robot.policy)
    
    # Create cost critic network for constrained RL
    cost_actor_critic = Policy(
        envs.observation_space.spaces,
        envs.action_space,
        env_config,
        base_kwargs=algo_args,
        base=config.robot.policy)

    # Initialize rollout storage buffer for collecting experience
    rollouts = RolloutStorage(algo_args.num_steps,
                              algo_args.num_processes,
                              envs.observation_space.spaces,
                              envs.action_space,
                              algo_args.human_node_rnn_size,
                              algo_args.human_human_edge_rnn_size)

    # Resume training from checkpoint if specified
    if algo_args.resume:
        raise NotImplementedError

    # Move networks to GPU if available
    nn.DataParallel(actor_critic).to(device)
    nn.DataParallel(cost_actor_critic).to(device)

    # Initialize PPO optimizer with or without cost constraints
    if env_config.policy.constrain_cost:
        # PPO with Lagrangian multipliers for cost constraints
        agent = ppo.PPOLag(
            actor_critic,
            cost_actor_critic,
            algo_args.clip_param,
            algo_args.ppo_epoch,
            algo_args.num_mini_batch,
            algo_args.value_loss_coef,
            algo_args.entropy_coef,
            cost_limit=env_config.constrained_rl_related.cost_limit,
            lag_init=env_config.constrained_rl_related.lag_init, 
            lag_lr=env_config.constrained_rl_related.lag_lr,
            lr=algo_args.lr,
            eps=algo_args.eps,
            max_grad_norm=algo_args.max_grad_norm,
            )
    else:
        # Standard PPO without cost constraints
        agent = ppo.PPOLagOriginalUpdate(
            actor_critic,
            cost_actor_critic,
            algo_args.clip_param,
            algo_args.ppo_epoch,
            algo_args.num_mini_batch,
            algo_args.value_loss_coef,
            algo_args.entropy_coef,
            lr=algo_args.lr,
            eps=algo_args.eps,
            max_grad_norm=algo_args.max_grad_norm)
        

    # Initialize Weights & Biases for experiment tracking
    trial_name = model_name
    wandb.init(project="robot_crowd_navigation", name=trial_name)
    # Define custom metrics for logging
    wandb.define_metric("env_step")
    wandb.define_metric("train_step")
    wandb.define_metric("env/*", step_metric="env_step")
    wandb.define_metric("train/*", step_metric="train_step")
    
    # Save experiment name to file
    note_file_path = os.path.join(algo_args.output_dir, 'note.txt')
    with open(note_file_path, 'w') as file:
        file.write(trial_name)

    # Initialize tracking variables
    step_iter = 0
    env_iter = 0
    train_iter = 0

    # Reset environment and get initial observations
    obs = envs.reset()
    # Get ACI (Adaptive Conformity Index) predictions from environment
    out_pred = obs['spatial_edges'][:, :, :].to('cpu').numpy()
    outs = envs.talk2Env(out_pred)
    aci_predicted_conformity_scores = np.array([o[0] for o in outs])  # [num_envs, num_humans, num_pred_steps]
    aci_cost = np.array([o[1] for o in outs])  # [num_envs,]
    obs['conformity_scores'] = torch.from_numpy(aci_predicted_conformity_scores).to(device)
    
    # Initialize rollout storage with first observations
    if isinstance(obs, dict):
        for key in obs:
            rollouts.obs[key][0].copy_(obs[key])
    else:
        rollouts.obs[0].copy_(obs)

    rollouts.to(device)

    # Initialize tracking queues for episode statistics
    episode_rewards = deque(maxlen=500)
    episode_collisions = deque(maxlen=500)
    episode_costs = deque(maxlen=500)
    episode_costs_for_updating_lagrange = deque(maxlen=32)  # For Lagrange multiplier updates
    episode_rewards_for_showing_rewards = deque(maxlen=32)
    best_score = -10000

    start = time.time()
    # Calculate total number of training updates
    num_updates = int(algo_args.num_env_steps) // algo_args.num_steps \
                                               // algo_args.num_processes

    # Main training loop
    for j in range(num_updates):
        # Schedule learning rate decay if enabled
        if algo_args.use_linear_lr_decay:
            network_utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if algo_args.algo == "acktr" else
                algo_args.lr)

        # Collect experience for num_steps timesteps
        for step in range(algo_args.num_steps):
            # Sample actions from current policy
            with torch.no_grad():
                # Prepare observations for both main and cost networks
                rollouts_obs = {}
                roullouts_obs_for_cost = {}
                for key in rollouts.obs:
                    rollouts_obs[key] = rollouts.obs[key][step]
                    roullouts_obs_for_cost[key] = rollouts.obs[key][step].clone()
                
                # Prepare hidden states for RNN networks
                rollouts_hidden_s = {}
                rollouts_hidden_s_for_cost = {}
                for key in rollouts.recurrent_hidden_states:
                    rollouts_hidden_s[key] = \
                        rollouts.recurrent_hidden_states[key][step]
                    rollouts_hidden_s_for_cost[key] = \
                        rollouts.recurrent_hidden_states[key][step].clone()
                
                # Get action from main policy
                value, action, action_log_prob, recurrent_hidden_states = \
                    actor_critic.act(rollouts_obs,
                                     rollouts_hidden_s,
                                     rollouts.masks[step])
                
                # Get cost value from cost critic
                cost_value, _, _, _ = \
                    cost_actor_critic.act(roullouts_obs_for_cost,
                                          rollouts_hidden_s_for_cost,
                                          rollouts.masks[step].clone())

            # Render environment if enabled
            if config.sim.render:
                envs.render()
            
            # Take action and observe results
            obs, reward, done, infos = envs.step(action)
            
            # Get ACI predictions and add noise for robustness
            out_pred = obs['spatial_edges'][:, :, :].to('cpu').numpy()
            outs = envs.talk2Env(out_pred)
            aci_predicted_conformity_scores = np.array([o[0] for o in outs])
            aci_cost = np.array([o[1] for o in outs])
            
            # Add Gaussian noise to conformity scores for data augmentation
            std_dev = env_config.aci_related.noise_std_for_conformity_scores
            gaussian_noise_predicted_conformity_scores = np.random.normal(0.0, std_dev, aci_predicted_conformity_scores.shape)

            # Clip noise to prevent extreme values
            noise_clip_for_c_s = env_config.aci_related.noise_clip_for_conformity_scores
            clipped_noise_predicted_conformity_scores = np.clip(gaussian_noise_predicted_conformity_scores, -noise_clip_for_c_s, noise_clip_for_c_s)

            # Apply noise to conformity scores
            aci_predicted_conformity_scores = aci_predicted_conformity_scores + clipped_noise_predicted_conformity_scores
            
            obs['conformity_scores'] = torch.from_numpy(aci_predicted_conformity_scores).to(device)
            
            # Add ACI cost to episode cost tracking
            for i, info in enumerate(infos):
                info['cost'] += aci_cost[i]
            
            # Update environment monitor with new observations
            obs, reward, done, infos = envs.update_monitor(({key: obs[key].cpu().numpy() for key in obs}, reward.numpy(), done, infos))
            processed_costs = torch.tensor([[infos[i]['cost']] for i in range(len(infos))])
            
            # Process episode completion and logging
            for info in infos:
                if 'episode' in info.keys():
                    # Track episode statistics
                    episode_rewards.append(info['episode']['r'])
                    episode_costs.append(info['episode']['c'])
                    episode_costs_for_updating_lagrange.append(info['episode']['c'])
                    episode_rewards_for_showing_rewards.append(info['episode']['r'])

                    # Save best model based on recent performance
                    mean_num_for_saving = 200
                    avg_score = np.mean(list(episode_rewards)[int(-1*mean_num_for_saving):])
                    if avg_score > best_score:
                        best_score = avg_score
                        
                        # Save main policy
                        save_path_best = os.path.join(algo_args.output_dir, 'best_model')
                        if not os.path.exists(save_path_best):
                            os.mkdir(save_path_best)
                        
                        # Save cost policy
                        cost_save_path_best = os.path.join(algo_args.output_dir, 'cost_best_model')
                        if not os.path.exists(cost_save_path_best):
                            os.mkdir(cost_save_path_best)

                        torch.save(actor_critic.state_dict(),
                                   os.path.join(save_path_best, 'PPO' + ".pt"))
                        torch.save(cost_actor_critic.state_dict(),
                                   os.path.join(cost_save_path_best, 'PPO_cost' + ".pt"))
                
                    # Track collision events
                    if str(info['info']) == 'Collision':
                        episode_collisions.append(1.0)
                    else:
                        episode_collisions.append(0.0)
                    
                    # Log environment metrics to wandb
                    env_iter += 1
                    wandb.log({
                        "env_step": env_iter,
                        "env/Collision": 1 if str(info['info']) == 'Collision' else 0,
                        "env/ReachGoal": 1 if str(info['info']) == 'Reaching goal' else 0,
                        "env/Timeout": 1 if str(info['info']) == 'Timeout' else 0,
                        "env/Episode_Rewards": info['episode']['r'],
                        "env/Episode_Costs": info['episode']['c'],
                    })
                                        
            # Create masks for episode termination handling
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            
            # Store experience in rollout buffer
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, cost_value, reward, processed_costs, masks, bad_masks)

        # Compute value estimates for the last state
        with torch.no_grad():
            rollouts_obs = {}
            for key in rollouts.obs:
                rollouts_obs[key] = rollouts.obs[key][-1]
            rollouts_hidden_s = {}
            for key in rollouts.recurrent_hidden_states:
                rollouts_hidden_s[key] = \
                    rollouts.recurrent_hidden_states[key][-1]
            
            # Get final value estimates
            next_value = actor_critic.get_value(
                rollouts_obs, rollouts_hidden_s,
                rollouts.masks[-1]).detach()
            cost_next_value = cost_actor_critic.get_value(
                rollouts_obs, rollouts_hidden_s,
                rollouts.masks[-1]).detach()

        # Compute returns and advantages using GAE
        rollouts.compute_returns(next_value,
                                 cost_next_value,
                                 algo_args.use_gae,
                                 algo_args.gamma,
                                 algo_args.gae_lambda,
                                 algo_args.use_proper_time_limits)

        # Perform policy update
        mean_ep_costs = np.mean(np.array(episode_costs_for_updating_lagrange))
        mean_ep_rewards = np.mean(np.array(episode_rewards_for_showing_rewards))
        value_loss, cost_value_loss, lag_factor, action_loss, dist_entropy, adv_targ_epoch, cost_adv_targ_epoch = agent.update(rollouts, mean_ep_costs)

        rollouts.after_update()
        
        # Log training metrics
        train_iter += 1
        wandb.log({
            "train_step": train_iter,
            "train/value_loss": value_loss,
            "train/cost_value_loss": cost_value_loss,
            "train/lag_factor": lag_factor,
            "train/action_loss": action_loss,
            "train/dist_entropy": dist_entropy,
            "train/adv_targ_epoch": adv_targ_epoch,
            "train/cost_adv_targ_epoch": cost_adv_targ_epoch,
            "train/mean_ep_costs": mean_ep_costs,
            "train/mean_ep_rewards": mean_ep_rewards
        })

        # Save model checkpoints periodically
        if j % algo_args.save_interval == 0 or j == num_updates - 1:
            # Save main policy checkpoints
            save_path = os.path.join(algo_args.output_dir, 'checkpoints')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            
            # Save cost policy checkpoints
            cost_save_path = os.path.join(algo_args.output_dir, 'cost_checkpoints')
            if not os.path.exists(cost_save_path):
                os.mkdir(cost_save_path)

            torch.save(actor_critic.state_dict(),
                       os.path.join(save_path, '%.5i' % j + ".pt"))
            
            torch.save(cost_actor_critic.state_dict(),
                       os.path.join(cost_save_path, '%.5i' % j + ".pt"))

        # Print training progress
        if j % algo_args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * algo_args.num_processes * algo_args.num_steps
            end = time.time()

            print(
                "Updates {}, num timesteps {}, FPS {} \n"
                "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, "
                "mean/median cost {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.mean(episode_costs),
                    np.median(episode_costs),
                    np.min(episode_rewards),
                    np.max(episode_rewards)
                )
            )
            print(f"Collision rate across last {len(episode_collisions)} episodes: {np.mean(episode_collisions):.2f}")

            # Save training progress to CSV
            df = pd.DataFrame({'misc/nupdates': [j],
                               'misc/total_timesteps': [total_num_steps],
                               'fps': int(total_num_steps / (end - start)),
                               'eprewmean': [np.mean(episode_rewards)],
                               'loss/policy_entropy': dist_entropy,
                               'loss/policy_loss': action_loss,
                               'loss/value_loss': value_loss})

            if os.path.exists(os.path.join(algo_args.output_dir, 'progress.csv')) and j > 20:
                df.to_csv(os.path.join(algo_args.output_dir, 'progress.csv'),
                          mode='a', header=False, index=False)
            else:
                df.to_csv(os.path.join(algo_args.output_dir, 'progress.csv'),
                          mode='w', header=True, index=False)

    envs.close()
    
if __name__ == '__main__':
    main()