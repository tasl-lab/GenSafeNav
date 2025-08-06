from collections import deque

from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, config,section):
        super().__init__(config,section)
        self.sensor_range = config.robot.sensor_range
        self.past_locations = deque(maxlen=config.cv_prediction_steps) 
        self.past_predictions = deque(maxlen=config.cv_prediction_steps)
        self.past_Vx = deque(maxlen=config.cv_prediction_steps)
        self.past_Vy = deque(maxlen=config.cv_prediction_steps)
        self.aggr_factor = 1.0 
        self.aggr_index= 0 

    def set_aggr_factor(self, aggr_factor):
        self.aggr_factor = aggr_factor

    def set_aggr_index(self, aggr_index):
        self.aggr_index = aggr_index

    def get_aggr_factor(self):
        return self.aggr_factor

    def get_aggr_index(self):
        return self.aggr_index

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action


    def actWithJointState(self,ob):
        action = self.policy.predict(ob)
        return action
