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

# Define a useful tuple (optional)
        
class Policy(nn.Module):
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
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.input = nn.Linear(self.observation_dim, self.hidden_size)
        self.action_net_fc1 = nn.Linear(self.hidden_size, 32)
        self.action_net_fc2 = nn.Linear(32, 16)
        self.action_net_fc3 = nn.Linear(16, self.action_dim)
        self.softmax = nn.Softmax(dim=0)
        
        self.value_net_fc1 = nn.Linear(self.hidden_size, 32)
        self.value_net_fc2 = nn.Linear(32, 8)
        self.value_net_fc3 = nn.Linear(8, 1)        
        
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
        inputs = self.input(state)
        
        out = F.relu(self.action_net_fc1(inputs))
        out = F.relu(self.action_net_fc2(out))
        out = self.action_net_fc3(out)
        action_prob = self.softmax(out)
    
        value = F.relu(self.value_net_fc1(inputs))
        value = F.relu(self.value_net_fc2(value))
        value = self.value_net_fc3(value)
        self.values.append(value)
        ########## END OF YOUR CODE ##########

        return action_prob

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
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########
        # calculate V from trajectory
        for r in self.rewards[::-1]:
            R = r + gamma*R
            returns.insert(0, R)
          
        # calculate loss
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / returns.std()
        
        loss = 0
        for logprob, val, r in zip(self.logp, self.values, returns):
            advantage = r - val.item()
            action_loss = -logprob * advantage
            value_loss = F.mse_loss(val.float(), r.float())
            loss += (action_loss + value_loss)
        
        ########## END OF YOUR CODE ##########
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.logp[:]
        del self.values[:]


def train(lr=0.01):
    '''
        Train the model using SGD (via backpropagation)
        TODO: In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode
    '''    
    
    # Instantiate the policy model and the optimizer
    model = Policy()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0
        # Uncomment the following line to use learning rate scheduler
        scheduler.step()
        
        # For each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        
        ########## YOUR CODE HERE (10-15 lines) ##########
        for t in range(9999):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward+=reward
            if done:
                break
        
        optimizer.zero_grad()
        loss = model.calculate_loss()
        loss.backward()
        optimizer.step()
        model.clear_memory()
        
        ########## END OF YOUR CODE ##########
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {:5f}'.format(i_episode, t, ep_reward, ewma_reward))

        # check if we have "solved" the problem
        if ewma_reward > env.spec.reward_threshold:
            torch.save(model.state_dict(), './preTrained/LunarLander_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, n_episodes=10):
    '''
        Test the learned model (no change needed)
    '''      
    model = Policy()
    
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

