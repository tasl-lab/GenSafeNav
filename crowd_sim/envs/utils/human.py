from collections import deque

from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState

from dt_aci.one_step_dtai import DtACI

import numpy as np

class Human(Agent):
    # see Agent class in agent.py for details!!!
    def __init__(self, config, section):
        super().__init__(config, section)
        self.isObstacle = False 
        self.id = None 
        self.observed_id = -1
        self.past_locations = deque(maxlen=config.cv_prediction_steps) 
        self.past_predictions = deque(maxlen=config.cv_prediction_steps)
        
        self.pred_horizon_aci = 5
        self.prediction_valid = False
        self.reset_aci(config.aci_related.alpha)
        
        self.predicted_conformity_scores = []
        self.true_conformity_scores = []
        
        self.last_prediction = None
        self.last_aci_predicted_conformity_score = None

    def reset_aci(self, alpha):
        self.last_prediction = None
        self.last_aci_predicted_conformity_score = None
        self.gt_locations_aci = [] # not conformity score
        self.predictions_aci = [] # not conformity score
        self.pred_error_aci_list = [DtACI(alpha=alpha, initial_pred=(i+1)/10) for i in range(self.pred_horizon_aci)] #tag
        # predict 1 step, 2 step, 3 step, 4 step, 5 step conformity score quantile

    def update_aci(self):
        if len(self.gt_locations_aci) == 0 or len(self.predictions_aci) == 0:
            return

        curr_pos = self.gt_locations_aci[-1]
        num_aci_pred_step = min(self.pred_horizon_aci, len(self.gt_locations_aci))
        
        for i in range(num_aci_pred_step):
            aci_predictor = self.pred_error_aci_list[i]
            past_ith_prediction = self.predictions_aci[-(i + 1)]
            
            predicted_curr_pos = past_ith_prediction[i+1]
            nonconformity_score = np.linalg.norm(curr_pos - predicted_curr_pos)
            aci_predictor.update_true_value(nonconformity_score)


    # ob: a list of observable states
    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    # ob: a joint state (ego agent's full state + other agents' observable states)
    def act_joint_state(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        action = self.policy.predict(ob)
        return action
