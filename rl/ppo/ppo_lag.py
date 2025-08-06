import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from .lagrange import Lagrange
from .ppo import PPO



class PPOLag():
    """ Class for the PPO optimizer """
    def __init__(self,
                 actor_critic,
                 cost_actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 cost_limit,
                 lag_init=0.0, 
                 lag_lr=9e-4,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                ):

        self.actor_critic = actor_critic
        self.cost_actor_critic = cost_actor_critic 

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.cost_optimizer = optim.Adam(cost_actor_critic.parameters(), lr=lr/2, eps=eps) # tag 05/06/2024
        
        lagrange_cfgs = {
            'cost_limit': cost_limit, 
            'lagrangian_multiplier_init': lag_init,
            'lambda_lr': lag_lr,
            'lambda_optimizer': 'Adam'
        }
        
        self._lagrange = Lagrange(**lagrange_cfgs)

    def update(self, rollouts, mean_ep_costs):
        # update lagrange paramter first
        # tag
        self._lagrange.update_lagrange_multiplier(mean_ep_costs)
        
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)
        
        cost_advantages = rollouts.cost_returns[:-1] - rollouts.cost_value_preds[:-1]
        cost_advantages = (cost_advantages - cost_advantages.mean()) / (
            cost_advantages.std() + 1e-5)

        value_loss_epoch = 0
        cost_value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        
        adv_targ_epoch = 0
        cost_adv_targ_epoch = 0

        for e in range(self.ppo_epoch): 
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, cost_advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, cost_value_preds_batch, return_batch, cost_return_batch, \
                       masks_batch, old_action_log_probs_batch, adv_targ, cost_adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)
                
                cost_values, _, _, _, _ = self.cost_actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                penalty = self._lagrange.lagrangian_multiplier.item()
                adv_targ = (adv_targ - penalty * cost_adv_targ) / (1 + penalty)
                
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                
                # backprog the combined loss
                self.optimizer.zero_grad()
                self.cost_optimizer.zero_grad()
                
                total_loss=value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef  # TODO: options in config?
                cost_value_loss = 0.5 * (cost_return_batch - cost_values).pow(2).mean()
                combined_loss = total_loss + cost_value_loss                     
                combined_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm) 
                self.optimizer.step()
                self.cost_optimizer.step()
                
                value_loss_epoch += value_loss.item()
                cost_value_loss_epoch += cost_value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                
                adv_targ_epoch += adv_targ.mean().item()
                cost_adv_targ_epoch += cost_adv_targ.mean().item()


        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        cost_value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        
        adv_targ_epoch /= num_updates
        cost_adv_targ_epoch /= num_updates
        

        return value_loss_epoch, cost_value_loss_epoch, penalty, action_loss_epoch, dist_entropy_epoch, adv_targ_epoch, cost_adv_targ_epoch
