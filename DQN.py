# Navid Hassan Zadeh for comp400 summer 2023 project
#
# This file contains the Agent, DQN, and Memory classes. 
# It contains all the tools needed to handle the agents as well as 
# the neural network and replay memory. In other words, the agents
# and their brains are in this file.
#

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import random
from env import Environment
import torch.optim as optim

#this class manages a list of all the agents.
class Agents:
    def __init__(self,init,env):
        self.agent_list=[]
        self.init = init
        size = init["world_size"]
        self.height, self.width = size,size 
        self.env = env
        self.ohe_states = init["one_hot_encoded_states"]
        self.walls = self.init["walls"]
        
    # to add an agent to the list.
    def add_agent(self,start,targets,agent_id):
        if (start in self.init["walls"]):
            raise "agent location or target coincides with the walls"
        if targets != None:
            if  (set(targets) & set(self.init["walls"])):
                raise "agent location or target coincides with the walls"
        self.agent_list.append(Agent(self.init,self.env,start,targets,agent_id))
        return self.agent_list

    # this funciton is used in cases when agents are supposed to have random start locations in every episode. This method makes sure the random locaition assigned to the agent is valid and usable.
    def choose_random_empty_block(self):
        free_blocks = []
        for i in range(self.width):
            for j in range(self.height):
                if (i,j) not in self.init["walls"]:
                    free_blocks.append((i,j))
        return random.choice(free_blocks)
        
    # this method resets all the agents in the environment for a new episode.
    def reset(self):
        for agent in self.agent_list:
            agent.reset()

    #this method gives a one hot encoded version of the current location of the agents to be used as inputs of the neural network
    def one_hot_encode(self,pos):
        one_hot_encoded = []
        for i in range(self.width):
            for j in range(self.height):
                if (i,j) == pos:
                    one_hot_encoded.append(1)
                else:
                    one_hot_encoded.append(0)
        return one_hot_encoded
    
    #this function gives a list of all the agents' states
    def get_nn_inputs(self,undone_agents):
        agent_inputs = []
        for agent in undone_agents:
            agent_input = []
            if self.ohe_states:
                # we add the locaiton, target, and other information of the agent to its current state
                agent_input.extend(self.one_hot_encode(agent.location))
                targets_ohe = [self.one_hot_encode(pos) for pos in agent.targets]
                agent_input.extend(np.array(targets_ohe).flatten())
            else:
                agent_input.extend(agent.location)
                #agent_input.extend(np.array(agent.targets).flatten())
                agent_input.extend(agent.targets[agent.mode])
                # for y in range(agent.mode_num):
                #     agent_input.append(agent.mode == y)

                #print(f"mode: {agent.mode} and input is {agent_input}")
                
                #agent_input.extend(np.array(self.walls).flatten())
            
            for agt in self.agent_list:  ##adding other agents' locations and information
                if agt != agent:
                    if self.ohe_states:
                        agent_input.extend(self.one_hot_encode(agt.location))
                        targets_ohe = [self.one_hot_encode(pos) for pos in agt.targets]
                        agent_input.extend(np.array(targets_ohe).flatten())
                        agent_input.append(agt.arrived)
                    else:
                        agent_input.extend(agt.location) # add location of other agents
                        #agent_input.extend(np.array(agt.targets).flatten())
                        agent_input.extend(agt.targets[agt.mode]) # add target of other agents
                        agent_input.append(agt.arrived)  # add whther other agents arrived to their target
                        # for y in range(agt.mode_num): # this adds the mode of the agnets as one hot encoded
                        #     agent_input.append(agt.mode == y)                 
            #print(agent_input)
            agent_input_np = np.array(agent_input)
            agent_input_final = Variable(torch.from_numpy(agent_input_np)).view(1,-1)
            agent_inputs.append(agent_input_final)
        #print(agent_inputs)
        #exit()
        return agent_inputs
    
    #this method gives the predictions for each agent
    def predictions(self,nn_inputs,undone_agents):
        if len(nn_inputs)!= len(undone_agents):
            raise "unequal number of actions and agents"
        predicts = []
        for i in range(len(undone_agents)):
            prediction = undone_agents[i].predict(nn_inputs[i])
            predicts.append(prediction)
        return predicts
    
    #this function decides using epsilon greedy method whether each agent takes a random action or if it uses the current predictions to take a "smarter" action in the current episode
    def greedy_actions(self,undone_agents,predictions,epsilon):
        if len(predictions)!= len(undone_agents):
            raise "unequal number of actions and agents"
        actions = []
        for i in range(len(undone_agents)):
            action = undone_agents[i].greedy_action(predictions[i],epsilon)
            actions.append(action)
        return actions
    
    #this function stores the experience into the memory of each agent, seperately
    def push_memories(self,undone_agents,nn_inputs,actions,next_nn_inputs,rewards):
        # put a test to make sure the lengths are equal
        for i in range(len(undone_agents)):
            undone_agents[i].memory.push(nn_inputs[i].data, actions[i], next_nn_inputs[i].data, rewards[i])
    
    #this function saves the weights of the neural network in a file
    def save_weights(self):
        for agent in self.agent_list:
            agent.save_weights()

    # this function loads the weights of the neural network to resume operation without having to re-train the agents
    def load_weihgts(self):
        for agent in self.agent_list:
            agent.load_weihgts()

#this class manages a single agent.
class Agent:
    def __init__(self,init,env,initial_location,targets,agent_id=-1):
        self.actions = [[-1,0],[1,0],[0,-1],[0,1],[0,0],[1,1],[1,-1],[-1,-1],[-1,1]]
        self.wall_blocks=init["walls"]
        self.avoid_wall = init["preknown_possible_actions"]
        self.optimimize_choice = init["optimizer adam(0) or rms(1)"]
        self.lr = init["learning_rate"]
        size = init["world_size"]
        self.height, self.width = size,size 
        self.init = init
        self.env = env
        self.mode = 0
        #targets.append(initial_location)
        self.mode_num = len(targets)
        self.heart_level = 0
        
        #getting parameters from init
        if init["with_action_of_staying_still"]:
            self.action_num = 5
        else:
            self.action_num = 4
        self.buffer_size = init["buffer_size"]
        self.batch_size = init["batch_size"]
        self.gamma = init["gamma"]
        nn_input_size = init["nn_input_size"]

        #setting up the agent-specific information
        self.agent_id = agent_id #each agent id assigned a unique id number.

        self.initial_location,self.targets  = self.add_to_env(initial_location,targets)
        
        self.location = self.initial_location
        
        #setting up the NN, optimizer, loss function and memory for replay
        # note that each agent has its own independent neural network
        self.model = DQN(nn_input_size, self.init["neural_network_hidden_layers"], self.action_num, hidden_unit)
        if self.optimimize_choice==1:
            self.optimizer = optim.RMSprop(self.model.parameters(), self.lr)
        elif self.optimimize_choice==0:
            self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.criterion = torch.nn.MSELoss()
        self.memory = ReplayMemory(self.buffer_size)

    #this function adds the agent to the environment
    def add_to_env(self,initial_location,targets):
        if initial_location == None:
            self.random_spawn = True
            initial_location = self.choose_random_empty_block([])
        else: 
            self.random_spawn = False
        
        if targets == None:
            self.random_target = True
            targets = [self.choose_random_empty_block([initial_location])]
        else: 
            self.random_target = False

        self.env.agent_list.append(self)

        return initial_location,targets
    
    # this funciton is used in cases when agents are supposed to have random start locations in every episode. This method makes sure the random locaition assigned to the agent is valid and usable.
    def choose_random_empty_block(self,occupied):
        free_blocks = []
        other_agent_positions = [agent.location for agent in self.env.agent_list]
        for i in range(self.width):
            for j in range(self.height):
                loc = (i,j)
                if loc not in self.init["walls"] and loc not in other_agent_positions and loc not in occupied:
                    free_blocks.append((i,j))

        return random.choice(free_blocks)

    # this function resetss the location of the agent for a new episode
    def reset(self):
        if self.random_spawn:
            self.location = self.choose_random_empty_block([])
        else:
            self.location = self.initial_location

        if self.random_target:
            self.targets = [self.choose_random_empty_block([self.location])]

        self.arrived = False
        self.mode = 0
        self.heart_level = 0

    #this fuction makes a prediction based on the state of the agent.
    def predict(self,state):
        return self.model(state)
    
    #this fuction prepares the input for the neural network
    def nn_input_prepare(self,cells):
        return Variable(torch.from_numpy(cells)).view(1,-1)

    # this function uses the epsilon greedy algorithm to choose an action for the agent
    def greedy_action(self,prediction,epsilon):
        possible_actions=[]
        for i in range(self.action_num):
            new_loc = tuple(sum(x) for x in zip(self.actions[i],self.location))
            if  new_loc[0] < 0 or new_loc[0] >= self.width or \
                new_loc[1] < 0 or new_loc[1] >= self.height or (new_loc in self.wall_blocks):
                pass
            else:
                possible_actions.append(i)
        prob = np.random.random()
        if (prob < epsilon):
            if self.avoid_wall==False:
                action = np.random.randint(0,self.action_num)
            else:
                action = np.random.choice(possible_actions)
        else: 
            if self.avoid_wall==False:
                action = np.argmax(prediction.data)
            else: 
                data = prediction.data.tolist()[0]
                action_val = data[0]
                action = 0
                for p in possible_actions:
                    if data[p] >= action_val:
                        action_val = data[p]
                        action = p 
        return action

    #this function gives a batch of the specified size from the memory for replay
    def get_batch_from_memory(self):
        transitions = self.memory.sample(self.batch_size)
        return Transition(*zip(*transitions))

    #this function uses a batch to trian the neural network
    def learn(self,batch,i):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.LongTensor(batch.action)).view(-1,1)
        new_state_batch = Variable(torch.cat(batch.new_state))
        reward_batch = Variable(torch.FloatTensor(batch.reward))
        non_final_mask = (reward_batch == -1)
        
        qval_batch = self.predict(state_batch) #we use the q function to get predictions of q values for all actions possible
        state_action_values = qval_batch.gather(1, action_batch) # we perform the gradiant descent only on the action we are considering
        with torch.no_grad(): #Get max_Q(S',a)
            newQ = self.predict(new_state_batch)
        maxQ = newQ.max(1)[0]
        y = reward_batch
        y[non_final_mask] += self.gamma * maxQ[non_final_mask]
        y = y.view(-1,1)
        print("Episode number : %s" % (i,), end='\r') #
        loss = self.criterion(state_action_values, y)
        self.optimizer.zero_grad() # Optimization
        loss.backward()
        for p in self.model.parameters():
            p.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    # these two fuctions save and load the weights of the neural network inside several files
    def save_weights(self):
        torch.save(self.model.state_dict(), f'agent_saved_weights/agent_{self.agent_id}_weights.pth')

    def load_weihgts(self):
        self.model.load_state_dict(torch.load(f'agent_saved_weights/agent_{self.agent_id}_weights.pth'))



# this creates the structure of the hidden layers of the neural network
class hidden_unit(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(hidden_unit, self).__init__()
        self.activation = activation
        self.nn = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        out = self.nn(x)
        out = self.activation(out)   
        return out
        
# this function creates the neural networks used in every agent
class DQN(nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels, unit = hidden_unit, activation = F.relu):
        super(DQN, self).__init__()
        assert type(hidden_layers) is list
        self.hidden_units = nn.ModuleList()
        self.in_channels = in_channels
        prev_layer = in_channels
        for hidden in hidden_layers:
            self.hidden_units.append(unit(prev_layer, hidden, activation))
            prev_layer = hidden
        self.final_unit = nn.Linear(prev_layer, out_channels)
    
    def forward(self, x):
        out = x.view(-1,self.in_channels).float()
        for unit in self.hidden_units:
            out = unit(out)
        out = self.final_unit(out)
        return out



Transition = namedtuple('Transition',
                        ('state', 'action', 'new_state', 'reward'))
# the following class creates the momory used for replay in the project
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)        

    