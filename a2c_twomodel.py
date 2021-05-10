# Spring 2021, IOC 5269 Reinforcement Learning
# HW2: REINFORCE with baseline and A2C

import gym
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler

use_cuda = torch.cuda.is_available()

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        
        # model structure
        self.fc1 = nn.Linear(self.observation_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.dp = nn.Dropout(p=0.5)
        
        self.rewards = []
        self.values = []
        
        
    def forward(self, state):
        out = self.fc1(state)
        out = self.dp(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
    def calculate_loss(self, gamma=0.99):
        R = 0
        rewards = self.rewards
        values = self.values
        returns = []
        for r in rewards[::-1]:
            R = r + R*gamma
            returns.insert(0, R)
            
        policy_loss = []
        for i in range(len(values)-1):
#            policy_loss.append((values[i] + rewards[i] - values[i+1])**2)
            policy_loss.append((returns[i] - values[i])**2)
            
        loss = sum(policy_loss)/len(policy_loss)
        return loss
    
    def clear_memory(self):
        del self.rewards[:]
        del self.values[:]
        
    
class Actor(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the shared layer(s), the action layer(s), and the value layer(s)
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Actor, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.fc1 = nn.Linear(self.observation_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, self.action_dim)
        self.softmax = nn.Softmax(dim=0)

        ########## END OF YOUR CODE ##########
        
        # action & reward memory
        self.logp = []
        self.rewards = []
        self.values = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        out = self.softmax(out)
        ########## END OF YOUR CODE ##########

        return out


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_prob = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()
        ########## END OF YOUR CODE ##########
        
        # save to action buffer
        self.logp.append(m.log_prob(action))

        return action.item()


    def calculate_loss(self, gamma=0.99):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """
        
        # Initialize the lists and variables
        R = 0
        logp = self.logp
        rewards = self.rewards
        values = self.values
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########
        # calculate V from trajectory
        for r in rewards[::-1]:
            R = r + gamma*R
            returns.insert(0, R)
        
        # calculate loss

        policy_loss = []
        for i in range(len(returns)):
            policy_loss.append((returns[i] - values[i].detach())*-logp[i])
        
        loss = sum(policy_loss)/len(policy_loss)
        ########## END OF YOUR CODE ##########
        
        return loss

    def clear_memory(self):
        # reset action buffer
        del self.logp[:]
        del self.rewards[:]
        del self.values[:]



def train(lr=0.01):
    '''
        Train the model using SGD (via backpropagation)
        TODO: In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode
    '''    
    
    # Instantiate the policy model and the optimizer
    actor = Actor()
    critic = Critic()
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
#    actor_scheduler = Scheduler.StepLR(actor_optimizer, step_size=100, gamma=0.9)
#    critic_scheduler = Scheduler.StepLR(critic_optimizer, step_size=100, gamma=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0
        # Uncomment the following line to use learning rate scheduler
#        actor_scheduler.step()
#        critic_scheduler.step()
#        
        # For each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        
        ########## YOUR CODE HERE (10-15 lines) ##########
        for t in range(9999):
            action = actor.select_action(state)
            value = critic(torch.from_numpy(state).type(torch.FloatTensor))
            state, reward, done, _ = env.step(action)
            actor.rewards.append(reward)
            critic.rewards.append(reward)
            actor.values.append(value)
            critic.values.append(value)
            ep_reward+=reward
            if done:
                break

        actor_optimizer.zero_grad()
        actor_loss = actor.calculate_loss()
        actor_loss.backward()
        actor_optimizer.step()
        
        critic_optimizer.zero_grad()
        critic_loss = critic.calculate_loss()
        critic_loss.backward()
        critic_optimizer.step()
        
        actor.clear_memory()
        critic.clear_memory()
        
        ########## END OF YOUR CODE ##########
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {:5f}'.format(i_episode, t, ep_reward, ewma_reward))

        # check if we have "solved" the cart pole problem
        if ewma_reward > env.spec.reward_threshold:
            torch.save(actor.state_dict(), './preTrained/LunarLander_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break
        
        if i_episode == 3500:
            torch.save(actor.state_dict(), './preTrained/LunarLander_{}.pth'.format(lr))
            break


def test(name, n_episodes=10):
    '''
        Test the learned model (no change needed)
    '''      
    model = Actor()
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 20  
    lr = 0.01
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    train(lr)
    test('LunarLander_0.01.pth')

