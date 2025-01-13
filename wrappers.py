import torch
import numpy as np


class DMCEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.keys = list(self.env.observation_space.keys())
        dim = 0
        for key in self.keys:
            if key == 'head_height': # account for humanoid head
                dim += 1
            else:
                dim += self.env.observation_space[key].shape[-1]
        self.observation_space = np.zeros(dim)
        self.action_space = self.env.action_space
        self.max_episode_length = 1000
        self.device = torch.device("cuda:0")

    def reset(self):
        ob, info = self.env.reset()
        ob = self.cast_obs(ob)
        return ob

    def step(self, actions):
        # actions = actions.cpu().numpy()
        next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
        next_obs = self.cast_obs(next_obs)
        dones = np.logical_or(terminated, truncated)
        timeout = torch.tensor(truncated).bool().to(self.device)
        success = torch.tensor(terminated).to(self.device)
        info_ret = {'time_outs': timeout, 'success': success}

        return next_obs, rewards, dones, info_ret
    
    def cast_obs(self, ob):
        obs = []
        for key in self.keys:
            if key == 'head_height':
                obs.append(np.asarray(ob[key]).reshape(1))
            else:
                obs.append(ob[key])
        ob = np.concatenate(obs, axis=-1)
        return ob