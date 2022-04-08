from overcooked_ai_py.agents.agent import Agent
import torch
from pathlib import Path
from stable_baselines3.common.utils import obs_as_tensor
from overcooked_ai_py.mdp.actions import Action
from a2c_ppo_acktr.model import Policy
from human_aware_rl.rllib.rllib import OvercookedMultiAgent
from model import CustomCNN

old_name_to_new_name = {
    "simple": "cramped_room",
    "unident_s": "asymmetric_advantages",
    "random1": "coordination_ring",
    "random0": "forced_coordination",
    "random3": "counter_circuit",
}

class ApagAgentNewVersion(Agent):
    def __init__(self, actor_critic: Policy, agent_index: int, featurize_fn):
        self.actor_critic = actor_critic
        self.agent_index = agent_index
        self.featurize = featurize_fn
        self.reset()
        
    def reset(self):
        if self.actor_critic.is_recurrent:
            # TODO add recurrent policy initial state
            pass
        else:
            self.rnn_hxs = torch.zeros(1, self.actor_critic.recurrent_hidden_state_size)
            
    def action_probabilities(self, state):
        # NOTE code for test
        obs = self.featurize(state, debug=False)
        my_obs = obs[self.agent_index]
        if not isinstance(my_obs, torch.Tensor):
            my_obs = obs_as_tensor(my_obs, next(self.actor_critic.parameters()).device)
            
        my_obs = my_obs.unsqueeze(0)
        my_obs = my_obs.permute(0, 3, 1, 2)[:, 0:20].float()
            
        _, feats, rnn_hxs = self.actor_critic.base(my_obs, rnn_hxs, None)
        
        dist = self.actor_critic.dist(feats)
        
        return dist.probs.cpu().numpy()
            
    def action(self, state):
        # NOTE code only for test
        obs = self.featurize(state)
        my_obs = obs[self.agent_index]
        if not isinstance(my_obs, torch.Tensor):
            my_obs = obs_as_tensor(my_obs, next(self.actor_critic.parameters()).device)
         
        my_obs = my_obs.unsqueeze(0)
        my_obs = my_obs.permute(0, 3, 1, 2)[:, 0:20].float()
        _, action, action_log_prob, rnn_hxs = self.actor_critic.act(my_obs, self.rnn_hxs, None)
        
        agent_action_info = {
            "action_probs": action_log_prob.exp(),
        }
        agent_action = Action.INDEX_TO_ACTION[action[0]]
        
        self.rnn_hxs = rnn_hxs
        
        return agent_action, agent_action_info
    
    
def get_apag_agent(save_path, agent_index):
    save_path = Path(save_path)
    config = torch.load(save_path / "config.pt")
    policy = torch.load(save_path / "policy.pt", map_location="cpu")[0]
    config["multi_agent_params"] = {
            "reward_shaping_factor" : 0,
            "reward_shaping_horizon" : 0,
            "use_phi" : 0,
            "bc_schedule" : 0,
        }
    
    if config["layout_name"] in old_name_to_new_name.keys():
        config["layout_name"] = old_name_to_new_name[config["layout_name"]]
        config["mdp_params"]["layout_name"] = config["layout_name"]
    
    env = OvercookedMultiAgent.from_config(config)
    featurize_fn = env.featurize_fn_map["ppo"]
    return ApagAgentNewVersion(policy, agent_index, featurize_fn)
    
    