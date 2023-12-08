# By Navid Hassan Zadeh for comp400 summer 2023 project
#
# This file is where the prgram should be run from. 
# It contains two main methods: train and test, for 
# training and testing the performace of agents in 
# the gridworld environment.
#

from torch.autograd import Variable
import numpy as np
from visualization import Visualize
import torch.optim as optim
import time
import torch
from DQN import *
from env import *

action_names = ["left","right","up","down","stay","downright","upright","upleft","downleft"]


#initializations and settings of the program
init = { 
        "representation_mode": 2,    # 0 -> robots; 1 -> dots and circles; 3 -> minecraft 
        "world_size":10,  # this is the width and height of the environment
        "undeterminisim_constant":0.15,   # this determins how often the actions are undeterministic
        "neural_network_hidden_layers":[250,250,250,250,250,250,250,250],   # this determins the layout of the neural network
        "preknown_possible_actions":True,   # if True, it will limit the choice of actions of agent if it is near a wall or borders of the environment. 
        "with_action_of_staying_still": False,  # if True, it adds the action of "not moving" to the previous four actions of moving up, down , right, and left.
        "one_hot_encoded_states":False,    # if True, the location of the agent is given as one-hot encoded as opposed to coordinates.
        "display_environment_in_last_episodes_of_training": 0,  # to disable, set it to -1. otherwise, for example, if set to 2, it will display the last 2 episodes in training.
        "total_iterations" : 1700,
        "max_episode_length": 200,
        "optimizer adam(0) or rms(1)":1,    # the choice of the optimizer employed in the program. If 1, RMS, if 0, Adam's optimizer.
        "learning_rate":3e-3,       # this is the learnig rate of the neural network
        "gamma" : 0.9,             # gamma affects the value of rewards gained according to time.
        "buffer_size" : 10000,     
        "batch_size" : 150,     # how many "experience"s are sampled from the memory for replay.
        "nn_input_size":9,    # this is the imput size of the neural network. It must be changed if the number of agents changes.
        "walls": [(i,5) for i in range(0,9)]  #here are the coordinates of the walls placed in the environment
        + [(6,i) for i in range(2,6)]
        # +[(6,i) for i in range(0,4)]
        #  + [(i,6) for i in range(5,10)]
    }

buffer = init["buffer_size"]
BATCH_SIZE = init["batch_size"]
gamma= init["gamma"]
episode_number = init["total_iterations"]
max_ep_number = init["max_episode_length"]

env = Environment(init)    # create the environment object. This will remain for the entire program.

all_agents = Agents(init,env)   #here we create an object of type Agents. All the agents that we craeate will be added through this.

# In the following part, we create the agents and specify their location and targets.
# If the start (initial location) of the agent is set to None, the program will assign a random location every episode. 

#agent_list = all_agents.add_agent(start=(0,5),targets=[(9,9)],agent_id=0)
agent_list = all_agents.add_agent(start=None,targets=[(9,0)],agent_id=0)
agent_list = all_agents.add_agent(start=None,targets=[(9,9)],agent_id=1)
#agent_list = all_agents.add_agent(start=(5,7),targets=[(5,0)],agent_id=2)

#agent_list = all_agents.add_agent(start=(0,5),targets=[(9,0)],agent_id=1)
#agent_list = all_agents.add_agent(start=(8,4),targets=[(0,4)],agent_id=3)
#agent_list = all_agents.add_agent(start=None,targets=[(2,0)],agent_id=3)

show_learning=init["total_iterations"]-init["display_environment_in_last_episodes_of_training"]  # for debugging purposes

# this method will train the agents
def train():
    vis = Visualize(init)  # we set up the visualization in case we may need to display the environment and agents during training for debugging purposes.
    vis.initialize()
    epsilon = 1   # this determins the exploration vs exploitation ratio during training. We start with the value 1, and gradually as more episodes are completed, this number will decrease to get very close to 0.
    

    for i in range(episode_number):
        all_agents.reset() # we reset the agents for a new episode. If an agent's start location is set to None, this will assign a random start location to that agent.
        undone_agents = [agent for agent in agent_list]    # here we store a list of all the agents which are still not at their target.
        step = 0
        
        while(undone_agents != []): #this while loop will produce the steps to complete the episode. It will end on two conditions. Either all agents arrived at their destination or we surpass the maximum steps allowed in an episode.
            # For debugging: this will display the training process in the last steps of training. It is disabled by default.
            if i >= show_learning: 
                print(f"step {i}: undone agents {[agent.agent_id for agent in undone_agents]} at {[agent.location for agent in undone_agents]}") ##
                vis.render(env,i)
                time.sleep(0.5)
            nn_inputs = all_agents.get_nn_inputs(undone_agents)  # we get the current state of all the undone agents present in the environment.
            predictions = all_agents.predictions(nn_inputs,undone_agents)  # we get the predictions for the future action of the undone agents.
            actions = all_agents.greedy_actions(undone_agents,predictions,epsilon)  # this will determin based on epsilon-greedy whether to take a random action or take the action suggested by the current predictions for each undone agent.
            if i > show_learning: #debugging: display during training. Disabled by default.
                print(f"action by {[agent.agent_id for agent in undone_agents]} is {[action_names[act] for act in actions]}") ##
            rewards = env.step(actions,undone_agents)  # All agents will take their corresponding steps simultaneously.
            next_nn_inputs = all_agents.get_nn_inputs(undone_agents) #this will return the state of the agents after taking their step
            step +=1
            all_agents.push_memories(undone_agents,nn_inputs,actions,next_nn_inputs,rewards)  # we store the experience into the memory.

            for j in range(len(undone_agents)):
                
                #here, we check if the memory has enough experience for learning.
                if (len(undone_agents[j].memory) < buffer):
                    if rewards[j] != -1: 
                        break
                    else:
                        continue

                batch = undone_agents[j].get_batch_from_memory() #  get a random batch from memory
                undone_agents[j].learn(batch,i)   # use the bactch to train the neural network
            if max_ep_number != -1:   #if max episode number is set to -1, we iterate untill all agents arrive at their destination
                if step >init["max_episode_length"]:
                    break
            new_list =[]

            # update the list of undone agents.
            for agent in undone_agents:
                if agent.arrived == False:
                    new_list.append(agent)

            undone_agents = [agent for agent in new_list] 

        if undone_agents == []:
            print(f"All agents arrived in episode {i}")
        # elif len(undone_agents)==1:  # for debugging
        #     print(f"current mode of agent 1: {undone_agents[0].mode}")
        # else: 
        #     print(f"current mode of agent 1: {undone_agents[0].mode}")
        #     print(f"current mode of agent 2: {undone_agents[1].mode}")
        if epsilon > 0.005:  # set a lower bound for how small the epsilon can get
            epsilon -= (1/episode_number)

## this method tests of the trained agents by displaying it on the screen
def test():
    vis = Visualize(init)
    vis.initialize()
    i = 0
    all_agents.reset()  # reset all agents
    undone_agents = [agent for agent in agent_list]  #set all the agents as undone at the beginning

    #we perform a single episode and display the environment after each time step
    while(undone_agents != []):
        print(f"step {i}: undone agents {[agent.agent_id for agent in undone_agents]} at {[agent.location for agent in undone_agents]}") ##
        vis.render(env,i)
        time.sleep(0.1)

        #similar to training, we get the state of agents, get our prediction, and using epsilon greedy select their actions and perform the step.
        #for more explanation refer to the trainig section above
        nn_inputs = all_agents.get_nn_inputs(undone_agents)
        predictions = all_agents.predictions(nn_inputs,undone_agents)
        actions = all_agents.greedy_actions(undone_agents,predictions,0)
        print(f"action by {[agent.agent_id for agent in undone_agents]} is {[action_names[act] for act in actions]}") ##
        rewards = env.step(actions,undone_agents) #get the reward of the previous action taken by the agents for each agent
        i += 1
        if (i > 40): # this is an arbirtary upper limit to stop the episode if it takes longer than a certain number of episodes. It can be changed depeding on the environment.
            print("Game lost; too many moves.")
            break
    
        new_list =[]
        for agent in undone_agents:
            if agent.arrived == False:
                new_list.append(agent)

        undone_agents = [agent for agent in new_list]


#train()

#all_agents.save_weights()   # we store the weight of all the agents' neural network in a file to reuse them if need be, without needing to perform a long training process again.
all_agents.load_weihgts()  # here we can load the weights from a file stored previously.


#we keep repeating the displayed testing of the program until user exits
while(True):
    test()
