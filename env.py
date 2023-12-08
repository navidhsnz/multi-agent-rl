# Navid Hassan Zadeh for comp400 summer 2023 project
#
# This file contains the environment class. It handles the gridworld
# environment and the agents' movements in it.
#
#

import numpy as np
import pygame
import sys
import os
import random
import operator
from time import sleep
from tqdm import tqdm
from torch.autograd import Variable
import torch

AGENT = [1,0,0]
TARGET = [0,0,1]

random.seed(0)

#class Environment. This class manages the environment (agent locations, targets, walls, etc) and handles steps of the agents as well as the reward fucntion.
class Environment:
    def __init__(self,init):

        size = init["world_size"]
        self.undeterministic = init["undeterminisim_constant"]
        self.agent_list = []
        self.actions = [[-1,0],[1,0],[0,-1],[0,1],[0,0],[1,1],[1,-1],[-1,-1],[-1,1]]
        self.action_names = ["left","right","up","down","stay","downright","upright","upleft","downleft"]

        self.height, self.width = size,size 

        self.wall_blocks=init["walls"] # we get the wall locations from the init argument
        
    # this is a helper function of the step fuction that makes sure a location is not repeated inside a location list. For more information, read the step function first.
    def not_repeated(self, loc, loc_list):
        i = 0
        for item in loc_list:
            if item == loc:
                i+=1
        if i >=2: return False
        else: return True

    #this is the fuction responsible for taking a step for all agents within the environment simultaenously
    def step(self, acts,undone_agents):
        successful_step=[0 for action in acts]
        acts_undetermin = []
        # first, we handle the undeterministic-ness of the environment. 
        # Here we change some of the actions based on the undeterminism 
        # constant to make the environment harder and "less predictable" 
        # to navigate for the agents.
        for action in acts:
            prob = np.random.random()
            if (prob > self.undeterministic):
                acts_undetermin.append(action)
            else:
                if action == 0:
                    ac = random.choice([8,7])
                    acts_undetermin.append(ac)
                elif action == 1:
                    ac = random.choice([6,5])
                    acts_undetermin.append(ac)
                elif action == 2:
                    ac = random.choice([7,6])
                    acts_undetermin.append(ac)
                elif action == 3:
                    ac = random.choice([5,8])
                    acts_undetermin.append(ac)
                elif action == 4:
                    acts_undetermin.append(action)
        
        if len(acts)!= len(undone_agents):
            raise "unequal number of actions and agents"
        
        new_locs=[]
        for i in range(len(acts)):
            # we get the next location of the agent here 
            new_loc = tuple(sum(x) for x in zip(self.actions[acts_undetermin[i]],undone_agents[i].location))
            # we check whether the new location is within bounds and doesn't coincide with a wall.
            if  new_loc[0] < 0 or new_loc[0] >= self.width or \
                new_loc[1] < 0 or new_loc[1] >= self.height or (new_loc in self.wall_blocks):
                new_loc = undone_agents[i].location
                successful_step[i] = 1 #hit wall
            new_locs.append(new_loc)
        
        #checking and removing overlaps or blockage by other agents' path or next location.
        # we need to do this because agents take their steps simultaneously and we need to make sure several agents don't go to the same locaiton at once.
        updated_new_locs = []
        for i in range(len(new_locs)):
            if new_locs[i] not in [agent.location for agent in self.agent_list] and self.not_repeated(new_locs[i],new_locs):
                updated_new_locs.append(new_locs[i])
                successful_step[i] = 3 #successful
            else:
                updated_new_locs.append(undone_agents[i].location)
                successful_step[i] = 2 #blockage or accident

        if 0 in successful_step:
            print(successful_step)
            sleep(3)
        for o in range(len(acts)):
            if acts[o]==4:
                successful_step[o]=4

        for i in range(len(undone_agents)): #update the locaiton of the agents
            undone_agents[i].location = updated_new_locs[i]

        # we call the reward function to determin the rewards.
        rewards = self.rewards_function(undone_agents,successful_step)
        done = True

        return rewards

    # this function determins the reward for each agent independetly evaluated on its last action
    def rewards_function(self,undone_agents,successful_step):
        rewards = []
        for i in range(len(undone_agents)):
            # the agent mode is reserved for times when an agent
            #  has more than one target. once it arrives at one of
            #  its targets, the agent mode increases by one, showing
            #  that now it pursues the next target.
            if (undone_agents[i].location == undone_agents[i].targets[undone_agents[i].mode]): 
                reward = 10 # if agents arrives at target, we give a reward of 10
                if undone_agents[i].mode == undone_agents[i].mode_num-1: # if agent arrived at its final target. set the arrived flag to True.
                    undone_agents[i].arrived=True   # this flag will show the agent already arrived to all its targets
                else:
                    undone_agents[i].mode += 1  # agnet pursues the next target
            # this gives a negative reward if the agnet goes to the target of another agent.
            # elif undone_agents[i].location in [p.targets[0] for p in undone_agents]:
            #     reward = -10
            #     undone_agents[i].arrived=False
            else:
                # here we can assign different rewards depending on different scenarios that happened: agent hit a wall, or agent overlapped with another agent, etc.
                # if successful_step[i] == 3:
                #     reward =  -0.5
                # elif successful_step[i] == 2:
                #     reward =  -2
                # elif successful_step[i] == 1:
                #     reward =  -1
                # elif successful_step[i] == 4:
                #     reward =  -0.9
                # else:
                reward =  -1 # this is the standard -1 reward for every action that the agent takes.
                undone_agents[i].arrived=False


            rewards.append(reward)

        return rewards
