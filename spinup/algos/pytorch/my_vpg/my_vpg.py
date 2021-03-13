import numpy as np
import torch
import torch.nn as nn
import gym
import time
from spinup.utils.logx import EpochLogger

class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.episode_returns = []
    
    def append(self, state, action, reward, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
    
    def fill_episode_returns(self, ret):
        num_to_fill = len(self.rewards) - len(self.episode_returns)
        self.episode_returns += [ret] * num_to_fill
 
    def get(self):
        return (torch.tensor(self.states, dtype=torch.float32), 
            torch.tensor(self.actions, dtype=torch.int32), 
            torch.tensor(self.rewards, dtype=torch.float32), 
            torch.tensor(self.next_states, dtype=torch.float32), 
            torch.tensor(self.episode_returns, dtype=torch.float32))

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = output_activation if i == len(sizes) - 2 else activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

def my_vpg(env_fn, seed=0, steps_per_epoch=4000, epochs=50, max_ep_len=1000,
        hidden_sizes=[32], lr=1e-2,
        logger_kwargs=dict()):
    """
    Random Walk

    (This is simply a uniform random walk!)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

    """

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    print("env.observation_space", env.observation_space)
    print("env.observation_space.shape", env.observation_space.shape)
    print("env.action_space", env.action_space)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n

    # Prepare for interaction with environment
    start_time = time.time()

    # Instantiate networks
    policy_net = mlp(sizes = [obs_dim] + hidden_sizes + [act_dim], output_activation=nn.Softmax)
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    print("policy_net")
    print(policy_net)

    # value_net = mlp(sizes = [obs_dim] + hidden_sizes + [1])
    # value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)
    # print("value_net")
    # print(value_net)

    def get_policy(o):
        probs = policy_net(torch.as_tensor(o, dtype=torch.float32))
        return torch.distributions.Categorical(probs=probs)
    
    def get_action(o):
        return get_policy(o).sample().item()

    def get_logp(o, a):
        return get_policy(o).log_prob(a)

    # def get_value(o):
    #     return value_net(torch.as_tensor(o, dtype=torch.float32))

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        o, ep_ret, ep_len = env.reset(), 0, 0

        buffer = Buffer()
        for t in range(steps_per_epoch):
            # Pick a random action within the action space
            if isinstance(env.action_space, gym.spaces.Box):
                raise NotImplementedError
                # a = np.random.uniform(env.action_space.low, env.action_space.high)
            elif isinstance(env.action_space, gym.spaces.Discrete):
                a = get_action(o)
                # a = np.random.randint(env.action_space.n)

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            buffer.append(o, a, r, next_o)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                buffer.fill_episode_returns(ep_ret)
                o, ep_ret, ep_len = env.reset(), 0, 0
        
        # Update
        o, a, r, next_o, R = buffer.get()

        # baseline = get_value(o)
        # R = r + get_value(next_o)
        # advantage = R - baseline

        # # Value function update
        # value_optimizer.zero_grad()
        # criterion = torch.nn.MSELoss()
        # value_loss = criterion(R, baseline)
        # value_loss.backward()
        # value_optimizer.step()

        # Policy function update
        policy_optimizer.zero_grad()
        logp_a = get_logp(o, a)
        policy_loss = -(logp_a * R).mean()
        policy_loss.backward()
        policy_optimizer.step()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='my_vpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    my_vpg(lambda : gym.make(args.env), seed=args.seed, 
        steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)