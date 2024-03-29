import gym
import custom_gym_envs

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

import numpy as np

import itertools
from copy import deepcopy
import math

import shared_env_updates


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.observation_dims = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.fc1 = nn.Linear(self.observation_dims, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, self.n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.out(x))

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.observation_dims = np.sum([box.shape[0] for box
                                        in env.observation_space.spaces])
        self.n_actions = env.action_space.n

        self.fc1 = nn.Linear(self.observation_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, self.n_actions)

    def forward(self, x):
        x = x.reshape([-1, self.observation_dims])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.out(x))

class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.observation_dims = np.sum([box.shape[0] for box
                                        in env.observation_space.spaces])
        self.n_actions = env.action_space.n

        self.fc1 = nn.Linear(self.observation_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = x.reshape([-1, self.observation_dims])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


class Forager(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.observation_dims = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.fc1 = nn.Linear(self.observation_dims, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, self.n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.out(x))


class SocialForager(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.observation_dims = np.sum([box.shape[0] for box
                                        in env.observation_space.spaces])
        self.n_actions = env.action_space.n

        self.fc1 = nn.Linear(self.observation_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, self.n_actions)

    def forward(self, x):
        x = x.reshape([-1, self.observation_dims])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.out(x))


def a2c(env, actor, critic, n_episodes, batch_size, gamma, lr):

    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

    # TODO: track the raw rewards as well for debugging purposes
    batch_rewards = []
    batch_states = []
    batch_actions = []
    batch_values = []
    batch_counter = 1

    for ep in range(n_episodes):
        total_rewards = []
        current_state = env.reset()
        states = []
        actions = []
        rewards = []
        values = []
        done = False
        while not done:
            action_probs = actor(torch.FloatTensor(current_state))
            action = np.random.choice(env.action_space.n,
                                      p=action_probs[0].detach().numpy())
            new_state, reward, done, _ = env.step(action)
            value = critic(torch.FloatTensor(current_state))
            values.append(value)
            states.append(current_state)
            actions.append(action)
            rewards.append(reward)

            current_state = new_state

        else:
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_rewards.extend(discount_rewards(rewards, gamma))
            batch_values.extend(values)
            batch_counter += 1
            total_rewards.append(sum(rewards))

        if batch_counter == batch_size:
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            state_tensor = torch.FloatTensor(batch_states)
            action_tensor = torch.LongTensor(batch_actions)
            reward_tensor = torch.FloatTensor(batch_rewards)
            value_tensor = critic(state_tensor)

            log_ps = torch.log(actor(state_tensor))

            action_log_ps = log_ps[np.arange(len(action_tensor)), action_tensor]

            advantages = reward_tensor - value_tensor

            action_log_ps_x_rewards = action_log_ps * advantages.detach()

            actor_loss = -action_log_ps_x_rewards.mean()
            critic_loss = advantages.pow(2).mean()

            actor_loss.backward()
            critic_loss.backward()

            actor_optimizer.step()
            critic_optimizer.step()

            # Print running average
            print("\rEp: {} Average of last 100: {:.2f}".format(
                ep + 1, np.mean(total_rewards[-100:])), end="")

            batch_rewards = []
            batch_states = []
            batch_actions = []
            batch_counter = 1

def a2c_multi(env_pol_list, n_episodes, batch_size, gamma, lr):

    env_optim_list = []

    n_dir_perms = math.factorial(len([- 1, 0, 1]))
    n_entity_perms = math.factorial(len(env_pol_list))

    for actor, critic, env in env_pol_list:
        actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
        critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
        env_optim_list.append(dict(actor=actor,
                                   critic=critic,
                                   actor_optimizer=actor_optimizer,
                                   critic_optimizer=critic_optimizer,
                                   env=env,
                                   batch_rewards=[],
                                   batch_states=[],
                                   batch_actions=[],
                                   batch_values=[],
                                   batch_counter=1,
                                   total_rewards=[],
                                   states=[],
                                   actions=[],
                                   rewards=[],
                                   values=[],
                                   done=False))



    for ep in range(n_episodes):

        if ep == 10000:
            print('stop')

        shared_env = shared_env_updates.init_plants(np.random.randint(2, 9))  # TODO: remove hard-coded stuff
        for agent in env_optim_list:
            agent['current_state'] = agent['env'].reset(shared_env)
            agent['done'] = False
            agent['states'] = []
            agent['actions'] = []
            agent['rewards'] = []
            agent['values'] = []
        agent_locs = [agent['env'].loc for agent in env_optim_list]
        for i, agent in enumerate(env_optim_list):
            own_locs = deepcopy(agent_locs)
            _ = own_locs.pop(i)
            agent['env'].add_others_locs(own_locs)
            agent['env'].id = i



        while not all([agent['done'] for agent in env_optim_list]):
            for agent in env_optim_list:
                if not agent['done']:
                    action_probs = agent['actor'](torch.FloatTensor(agent['current_state']))
                    agent['action'] = np.random.choice(agent['env'].action_space.n,
                                              p=action_probs[0].detach().numpy())
                    value = agent['critic'](torch.FloatTensor(agent['current_state']))
                    agent['actions'].append(agent['action'])
                    agent['values'].append(value)
                    agent['states'].append(agent['current_state'])


            # TODO: how to handle dead agents?
            entity_perm = np.random.randint(n_entity_perms)
            dir_perms = [[np.random.randint(n_dir_perms), np.random.randint(n_dir_perms)] for _ in env_optim_list]
            shared_env = shared_env_updates.step_vegetation(shared_env)
            actions = [[agent['action'] for agent in env_optim_list], dir_perms, entity_perm, deepcopy(shared_env)]

            for i, agent in enumerate(env_optim_list):
                agent_actions = deepcopy(actions)
                for j in range(2):
                    own = agent_actions[j].pop(i)
                    agent_actions[j] = [own, *agent_actions[j]]
                if not agent['done']:
                    new_state, reward, done, _ = agent['env'].step(agent_actions)
                    agent['rewards'].append(reward)
                    agent['done'] = done
                    agent['current_state'] = new_state
                shared_env = agent['env'].plants
        else:
            for agent in env_optim_list:
                agent['batch_states'].extend(agent['states'])
                agent['batch_actions'].extend(agent['actions'])
                agent['batch_rewards'].extend(discount_rewards(agent['rewards'], gamma))
                agent['batch_values'].extend(agent['values'])
                agent['batch_counter'] += 1
                agent['total_rewards'].append(sum(agent['rewards']))

        for agent in env_optim_list:
            if agent['batch_counter'] == batch_size:
                agent['actor_optimizer'].zero_grad()
                agent['critic_optimizer'].zero_grad()
                state_tensor = torch.FloatTensor(agent['batch_states'])
                action_tensor = torch.LongTensor(agent['batch_actions'])
                reward_tensor = torch.FloatTensor(agent['batch_rewards'])
                value_tensor = agent['critic'](state_tensor)

                log_ps = torch.log(agent['actor'](state_tensor))

                action_log_ps = log_ps[np.arange(len(action_tensor)), action_tensor]

                advantages = reward_tensor - value_tensor

                action_log_ps_x_rewards = action_log_ps * advantages.detach()

                actor_loss = -action_log_ps_x_rewards.mean()
                critic_loss = advantages.pow(2).mean()

                actor_loss.backward()
                critic_loss.backward()

                agent['actor_optimizer'].step()
                agent['critic_optimizer'].step()

                # Print running average
                print("\rEp: {} Average of last 100: {:.2f}".format(
                    ep + 1, np.mean(agent['total_rewards'][-100:])), end="")

                agent['batch_rewards'] = []
                agent['batch_states'] = []
                agent['batch_actions'] = []
                # TODO: should I be clearing the batch values as well?
                agent['batch_counter'] = 1


def reinforce(env, pol, n_episodes, batch_size, gamma, lr):

    optimizer = optim.RMSprop(pol.parameters(), lr=lr)

    # TODO: track the raw rewards as well for debugging purposes
    batch_rewards = []
    batch_states = []
    batch_actions = []
    batch_counter = 1

    for ep in range(n_episodes):
        if ep == 2000:
            print('stop')
        total_rewards = []
        current_state = env.reset()
        states = []
        actions = []
        rewards = []
        done = False
        while not done:
            action_probs = pol(torch.FloatTensor(current_state))
            action = np.random.choice(env.action_space.n,
                                      p=action_probs[0].detach().numpy())
            new_state, reward, done, _ = env.step(action)
            states.append(current_state)
            actions.append(action)
            rewards.append(reward)

            current_state = new_state

        else:
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_rewards.extend(discount_rewards(rewards, gamma))
            batch_counter += 1
            total_rewards.append(sum(rewards))

        if batch_counter == batch_size:
            optimizer.zero_grad()
            state_tensor = torch.FloatTensor(batch_states)
            action_tensor = torch.LongTensor(batch_actions)
            reward_tensor = torch.FloatTensor(batch_rewards)

            log_ps = torch.log(pol(state_tensor))

            action_log_ps = log_ps[np.arange(len(action_tensor)), action_tensor]

            action_log_ps_x_rewards = action_log_ps * reward_tensor

            loss = -action_log_ps_x_rewards.mean()

            loss.backward()
            optimizer.step()

            # Print running average
            print("\rEp: {} Average of last 100: {:.2f}".format(
                ep + 1, np.mean(total_rewards[-100:])), end="")

            batch_rewards = []
            batch_states = []
            batch_actions = []
            batch_counter = 1


def test_gridworld(env, policy, n_episodes):
    for ep in range(n_episodes):
        print('BEGINING NEW EPISODE:')
        done = False
        ep_reward = 0
        state = env.reset()
        while not done:
            action_probs = policy(torch.FloatTensor(state))
            action = np.random.choice(env.action_space.n,
                                      p=action_probs[0].detach().numpy())
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            print('loc: {0}'.format(state[0:2]))
        print('Episode reward: {0}'.format(ep_reward))


def test_forager(env, policy, n_episodes):
    action_names = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}
    print("\n\nMAP OF FORAGE WORLD:")
    for y in reversed(range(7)):
        row = []
        for x in range(7):
            row.append(env._plants.get((x, y), 0) + env._mushrooms.get((x, y), 0))
        print(row)
    for ep in range(n_episodes):
        print('BEGINING NEW EPISODE:')
        done = False
        ep_reward = 0
        state = env.reset()
        print('Initial location of agent: {0}'.format(env._loc))
        while not done:
            print('Agent HP: {0}'.format(env._hp))
            print('Observation:')
            print('{0}\n{1}\n{2}'.format(state[6:], state[3:6], state[:3]))
            action_probs = policy(torch.FloatTensor(state))
            action = np.argmax(action_probs.detach().numpy())
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            print('Action: {}'.format(action_names[action]))
        print('Episode reward: {0}'.format(ep_reward))


def test_social_forager(env, policy, n_episodes):
    action_names = {0: 'Up',
                    1: 'Right',
                    2: 'Down',
                    3: 'Left',
                    4: 'Attack',
                    5: 'Share'}
    print("\n\nMAP OF FORAGE WORLD:")
    for y in reversed(range(7)):
        row = []
        for x in range(7):
            row.append(env._plants.get((x, y), 0) + env._mushrooms.get((x, y), 0))
        print(row)
    ep_rewards = []
    for ep in range(n_episodes):
        print('BEGINING NEW EPISODE:')
        done = False
        ep_reward = 0
        state = env.reset()
        print('Initial location of agent: {0}'.format(env._loc))
        print('Initial location of foe: {0}'.format(env._foes[0]._loc))
        print('Initial location of friend: {0}'.format(env._friends[0]._loc))
        while not done:
            print("\n\nMAP OF FORAGE WORLD:")
            for y in reversed(range(7)):
                row = []
                for x in range(7):
                    row.append(
                        env._plants.get((x, y), 0) + env._mushrooms.get((x, y), 0))
                print(row)
            print('Agent location: {0}'.format(env._loc))
            print('Foe location: {0}'.format(env._foes[0]._loc))
            print('Friend location: {0}'.format(env._friends[0]._loc))
            print('Agent HP: {0}'.format(env._hp))
            print('Vegetation view:')
            print('{0}\n{1}\n{2}'.format(state[0][6:], state[0][3:6], state[0][:3]))
            print('Foe view:')
            print('{0}\n{1}\n{2}'.format(state[3][6:], state[3][3:6], state[3][:3]))
            print('Friend view:')
            print('{0}\n{1}\n{2}'.format(state[4][6:], state[4][3:6], state[4][:3]))
            action_probs = policy(torch.FloatTensor(state))
            action = np.argmax(action_probs.detach().numpy())
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            print('Action: {}'.format(action_names[action]))
        print('Episode reward: {0}'.format(ep_reward))
        ep_rewards.append(ep_reward)
    print('Mean Episode reward: {0}'.format(np.mean(ep_rewards)))
    print('StDev Episode reward: {0}'.format(np.std(ep_rewards)))


def test_multi_forager(env_pol_list, n_episodes):
    action_names = {0: 'Up',
                    1: 'Right',
                    2: 'Down',
                    3: 'Left',
                    4: 'Attack',
                    5: 'Share'}

    agent_list = []
    n_dir_perms = math.factorial(len([- 1, 0, 1]))
    n_entity_perms = math.factorial(len(env_pol_list))

    for actor, _, env in env_pol_list:
        agent = dict(env=env,
                     pol=actor,
                     done=False,
                     state=None,
                     action=None,
                     reward=0,
                     ep_reward=0)
        agent_list.append(agent)

    ep_rewards = [[], []]
    for ep in range(n_episodes):
        print('BEGINING NEW EPISODE:')
        shared_env = shared_env_updates.init_plants(np.random.randint(2, 9))  # TODO: remove hard-coded stuff

        for agent in agent_list:
            agent['state'] = agent['env'].reset(shared_env)
            agent['done'] = False
            agent['ep_reward'] = 0
        agent_locs = [agent['env'].loc for agent in agent_list]

        for i, agent in enumerate(agent_list):
            own_locs = deepcopy(agent_locs)
            _ = own_locs.pop(i)
            agent['env'].add_others_locs(own_locs)
            agent['env'].id = i


        while not all([agent['done'] for agent in agent_list]):
            entity_perm = np.random.randint(n_entity_perms)
            dir_perms = [[np.random.randint(n_dir_perms), np.random.randint(n_dir_perms)] for _ in agent_list]
            shared_env = shared_env_updates.step_vegetation(shared_env)
            print("\n\nMAP OF FORAGE WORLD:")
            rows = []
            for y in reversed(range(7)):
                row = []
                for x in range(7):
                    row.append(
                        shared_env.get((x, y), 0))
                rows.append(row)

            symbols = 'XY'
            hps = []
            for i, agent in enumerate(agent_list):
                if not agent['done']:
                    rows[-agent['env'].loc[1]-1][agent['env'].loc[0]] = symbols[i]
                hps.append(agent['env'].hp)
            for row in rows:
                print(row)
            for i, agent in enumerate(agent_list):
                print('Agent {0} HP: {1}'.format(symbols[i], hps[i]))

            actions = []
            for i, agent in enumerate(agent_list):
                action_probs = agent['pol'](torch.FloatTensor(agent['state']))
                action = np.argmax(action_probs.detach().numpy())
                actions.append(action)
                if not agent['done']:
                    print('{0} Action: {1}'.format(symbols[i], action_names[action]))
                else:
                    print('{0} Action: None (Dead)'.format(symbols[i]))

            actions = [actions, dir_perms, entity_perm, deepcopy(shared_env)]

            for i, agent in enumerate(agent_list):
                agent_actions = deepcopy(actions)
                for j in range(2):
                    own = agent_actions[j].pop(i)
                    agent_actions[j] = [own, *agent_actions[j]]
                if not agent['done']:
                    new_state, reward, done, _ = agent['env'].step(agent_actions)
                    agent['reward'] = reward
                    agent['done'] = done
                    agent['state'] = new_state
                    agent['ep_reward'] += reward
                    shared_env = agent['env'].plants


        for i, agent in enumerate(agent_list):
            print('{0} Episode reward: {1}'.format(symbols[i], agent['ep_reward']))
            ep_rewards[i].append(agent['ep_reward'])

    for i, agent in enumerate(agent_list):
        print('{0} Mean Episode reward: {1}'.format(symbols[i], np.mean(ep_rewards[i])))
        print('{0} StDev Episode reward: {1}'.format(symbols[i], np.std(ep_rewards[i])))


def discount_rewards(reward_list, gamma):
    r = np.array([gamma ** i * reward for i, reward
                  in enumerate(list(reward_list))])

    r = np.flip(r)
    r = np.cumsum(r)
    r = np.flip(r)

    r = r - r.mean()

    return r


if __name__ == '__main__':
    ENV_NAME = 'hunters-gatherers-multi-v0'

    # I think either .01 or .001? is a good lr for gridworld

    world1 = gym.make(ENV_NAME)
    world2 = gym.make(ENV_NAME)

    #policy = SocialForager(world)

    actor1 = Actor(world1)
    critic1 = Critic(world1)

    actor2 = Actor(world2)
    critic2 = Critic(world2)

    env_pol_list = [(actor1, critic1, world1), (actor2, critic2, world2)]

    a2c_multi(env_pol_list,
              n_episodes=50000,
              batch_size=200,
              gamma=0.99,
              lr=0.001)

    test_multi_forager(env_pol_list, n_episodes=10)

