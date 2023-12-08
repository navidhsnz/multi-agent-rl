# Written by Navid Hassan Zadeh for comp400 summer 2023 project
#
# This file contains all the methods and tools
# needed to display an istance of the gridworld evironment
# along with all the objects in it such as: walls, agents, targets
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

WIDTH = 60
HEIGHT = 60
MARGIN = 1

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
ORANGE = (255, 128, 0)

# this is the main and only class in this file.
# It contains all the methods needed to display
# a visualization of the current status of the 
# gradworld environment.
class Visualize:
    
    def __init__(self,init):
        self.grid_size = init["world_size"]
        self.representation_mode = init["representation_mode"]
        self.init = init
        
    # this method initializes the object. It must be called before the any rendering takes place
    def initialize(self):
        pygame.init()
        pygame.display.set_caption('Visualization')
        self.my_font = pygame.font.SysFont("monospace", 30)
        self.clock = pygame.time.Clock()

        # we set the size of the window 
        board_size_x = (WIDTH + MARGIN) * self.grid_size
        board_size_y = (HEIGHT + MARGIN) * self.grid_size
        window_size_x = int(board_size_x)
        window_size_y = int(board_size_y * 1.2)
        window_size = [window_size_x, window_size_y]
        self.screen = pygame.display.set_mode(window_size)

        # all the images used to signify agents, targets and walls are imported here.
        current_path = os.path.dirname(__file__)
        image_path = os.path.join(current_path, 'images')
         
         # the representation mode determins what set of images we 
         # want to use to display the environment. There are three 
         # options: robots, dots and circles, and minecraft mode.
         # this mode can be changed in the init dictionary in main.py
         #
        if self.representation_mode == 0:
            self.robots_load(image_path) 
        elif self.representation_mode==1:
            self.dot_and_circles_load(image_path)
        elif self.representation_mode==2:
            self.minecraft_load(image_path)

    #this method loads all the images in the minecraft folder.
    def minecraft_load(self,current_path):
        image_path = os.path.join(current_path, 'minecraft')
        
        img = pygame.image.load(os.path.join(image_path, 'wall.jpg')).convert()
        self.img_wall = pygame.transform.scale(img, (WIDTH, WIDTH))
        self.h_ = []
        img = pygame.image.load(os.path.join(image_path, '1h_blue.png')).convert()
        self.h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, '1h_green.png')).convert()
        self.h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, '1h_red.png')).convert()
        self.h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, '1h_pink.png')).convert()
        self.h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.c_ = []    
        img = pygame.image.load(os.path.join(image_path, 'c_blue.png')).convert()
        self.c_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c_green.png')).convert()
        self.c_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c_red.png')).convert()
        self.c_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c_pink.png')).convert()
        self.c_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.c1h_=[]
        img = pygame.image.load(os.path.join(image_path, 'c1h_blue.png')).convert()
        self.c1h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c1h_green.png')).convert()
        self.c1h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c1h_red.png')).convert()
        self.c1h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c1h_pink.png')).convert()
        self.c1h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.c2h_=[]
        img = pygame.image.load(os.path.join(image_path, 'c2h_blue.png')).convert()
        self.c2h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c2h_green.png')).convert()
        self.c2h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c2h_red.png')).convert()
        self.c2h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c2h_pink.png')).convert()
        self.c2h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.c3h_ =[]
        img = pygame.image.load(os.path.join(image_path, 'c3h_blue.png')).convert()
        self.c3h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c3h_green.png')).convert()
        self.c3h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c3h_red.png')).convert()
        self.c3h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c3h_pink.png')).convert()
        self.c3h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.m_ = []
        img = pygame.image.load(os.path.join(image_path, 'm_blue.png')).convert()
        self.m_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'm_green.png')).convert()
        self.m_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'm_red.png')).convert()
        self.m_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'm_pink.png')).convert()
        self.m_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.m1h_ = []
        img = pygame.image.load(os.path.join(image_path, 'm1h_blue.png')).convert()
        self.m1h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'm1h_green.png')).convert()
        self.m1h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'm1h_red.png')).convert()
        self.m1h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'm1h_pink.png')).convert()
        self.m1h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.m2h_ =[]
        img = pygame.image.load(os.path.join(image_path, 'm2h_blue.png')).convert()
        self.m2h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'm2h_green.png')).convert()
        self.m2h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'm2h_red.png')).convert()
        self.m2h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'm2h_pink.png')).convert()
        self.m2h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.m3h_ = []
        img = pygame.image.load(os.path.join(image_path, 'm3h_blue.png')).convert()
        self.m3h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'm3h_green.png')).convert()
        self.m3h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'm3h_red.png')).convert()
        self.m3h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'm3h_pink.png')).convert()
        self.m3h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.mc_ =[]
        img = pygame.image.load(os.path.join(image_path, 'mc_blue.png')).convert()
        self.mc_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'mc_green.png')).convert()
        self.mc_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'mc_red.png')).convert()
        self.mc_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'mc_pink.png')).convert()
        self.mc_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.mc1h_ =[]
        img = pygame.image.load(os.path.join(image_path, 'mc1h_blue.png')).convert()
        self.mc1h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'mc1h_green.png')).convert()
        self.mc1h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'mc1h_red.png')).convert()
        self.mc1h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'mc1h_pink.png')).convert()
        self.mc1h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.mc2h_ =[]
        img = pygame.image.load(os.path.join(image_path, 'mc2h_blue.png')).convert()
        self.mc2h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'mc2h_green.png')).convert()
        self.mc2h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'mc2h_red.png')).convert()
        self.mc2h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'mc2h_pink.png')).convert()
        self.mc2h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.mc3h_ =[]
        img = pygame.image.load(os.path.join(image_path, 'mc3h_blue.png')).convert()
        self.mc3h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'mc3h_green.png')).convert()
        self.mc3h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'mc3h_red.png')).convert()
        self.mc3h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'mc3h_pink.png')).convert()
        self.mc3h_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))

    #this method loads all the images in the dots and circles folder.
    def dot_and_circles_load(self,current_path):
        image_path = os.path.join(current_path, 'dot_and_circles') 

        img = pygame.image.load(os.path.join(image_path, 'wall.jpg')).convert()
        self.img_wall = pygame.transform.scale(img, (WIDTH, WIDTH))
        self.c_ = []
        img = pygame.image.load(os.path.join(image_path, 'c_black.png')).convert()
        self.c_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c_blue.png')).convert()
        self.c_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c_green.png')).convert()
        self.c_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c_lightblue.png')).convert()
        self.c_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c_orange.png')).convert()
        self.c_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c_pink.png')).convert()
        self.c_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'c_red.png')).convert()
        self.c_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.co_ = []
        img = pygame.image.load(os.path.join(image_path, 'co_black.png')).convert()
        self.co_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'co_blue.png')).convert()
        self.co_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'co_green.png')).convert()
        self.co_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'co_lightblue.png')).convert()
        self.co_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'co_orange.png')).convert()
        self.co_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'co_pink.png')).convert()
        self.co_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'co_red.png')).convert()
        self.co_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.d_ = []
        img = pygame.image.load(os.path.join(image_path, 'd_black.png')).convert()
        self.d_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'd_blue.png')).convert()
        self.d_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'd_green.png')).convert()
        self.d_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'd_lightblue.png')).convert()
        self.d_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'd_orange.png')).convert()
        self.d_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'd_pink.png')).convert()
        self.d_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'd_red.png')).convert()
        self.d_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.dc_ = []
        img = pygame.image.load(os.path.join(image_path, 'dc_black.png')).convert()
        self.dc_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'dc_blue.png')).convert()
        self.dc_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'dc_green.png')).convert()
        self.dc_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'dc_lightblue.png')).convert()
        self.dc_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'dc_orange.png')).convert()
        self.dc_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'dc_pink.png')).convert()
        self.dc_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'dc_red.png')).convert()
        self.dc_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.dco_ = []
        img = pygame.image.load(os.path.join(image_path, 'dco_black.png')).convert()
        self.dco_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'dco_blue.png')).convert()
        self.dco_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'dco_green.png')).convert()
        self.dco_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'dco_lightblue.png')).convert()
        self.dco_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'dco_orange.png')).convert()
        self.dco_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'dco_pink.png')).convert()
        self.dco_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'dco_red.png')).convert()
        self.dco_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.do_ = []
        img = pygame.image.load(os.path.join(image_path, 'do_black.png')).convert()
        self.do_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'do_blue.png')).convert()
        self.do_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'do_green.png')).convert()
        self.do_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'do_lightblue.png')).convert()
        self.do_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'do_orange.png')).convert()
        self.do_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'do_pink.png')).convert()
        self.do_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'do_red.png')).convert()
        self.do_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        self.o_ = []
        img = pygame.image.load(os.path.join(image_path, 'o_black.png')).convert()
        self.o_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'o_blue.png')).convert()
        self.o_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'o_green.png')).convert()
        self.o_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'o_lightblue.png')).convert()
        self.o_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'o_orange.png')).convert()
        self.o_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'o_pink.png')).convert()
        self.o_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'o_red.png')).convert()
        self.o_.append(pygame.transform.scale(img, (WIDTH, WIDTH)))

    #this method loads all the images in the robots folder.
    def robots_load(self,current_path):
        image_path = os.path.join(current_path, 'robots') 

        img = pygame.image.load(os.path.join(image_path, 'right.jpg')).convert()
        self.img_right = pygame.transform.scale(img, (WIDTH, WIDTH))
        img = pygame.image.load(os.path.join(image_path, 'left.jpg')).convert()
        self.img_left = pygame.transform.scale(img, (WIDTH, WIDTH))
        img = pygame.image.load(os.path.join(image_path, 'up.jpg')).convert()
        self.img_up = pygame.transform.scale(img, (WIDTH, WIDTH))
        img = pygame.image.load(os.path.join(image_path, 'down.jpg')).convert()
        self.img_down = pygame.transform.scale(img, (WIDTH, WIDTH))
        img = pygame.image.load(os.path.join(image_path, 'wall.jpg')).convert()
        self.img_wall = pygame.transform.scale(img, (WIDTH, WIDTH))
        img = pygame.image.load(os.path.join(image_path, 'target.jpg')).convert()
        self.img_target = pygame.transform.scale(img, (WIDTH, WIDTH))
        
        self.img_robots = []
        img = pygame.image.load(os.path.join(image_path, 'robot_1.jpg')).convert()
        self.img_robots.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'robot_2.jpeg')).convert()
        self.img_robots.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'robot_3.jpeg')).convert()
        self.img_robots.append(pygame.transform.scale(img, (WIDTH, WIDTH)))
        img = pygame.image.load(os.path.join(image_path, 'robot_4.png')).convert()
        self.img_robots.append(pygame.transform.scale(img, (WIDTH, WIDTH)))

    #this method fuction determins if the given position is a target block of any agent
    def is_target_block(self,pos,env):
        for agent in env.agent_list:
            if pos in agent.targets:
                return True
        return False
    
    #same as before but it doesn't consider targets that have been achieved
    def is_target_block_v2(self,pos,env):
        for agent in env.agent_list:
            if pos in agent.targets[agent.mode:]:
                return True
        return False

    # this function puts the targets of the dot-circle mode of visualization on the display
    def put_targets_dot_circles(self,pos,env):
        column, row = pos
        for agent in env.agent_list:
            for i in range(len(agent.targets)):
                if pos == agent.targets[i] and i == 0:
                    self.screen.blit(self.c_[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                if pos == agent.targets[i] and i == 1:
                    self.screen.blit(self.o_[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
 
    # this function puts the unachieved targets of the dot-circle mode of visualization on the display
    def put_targets_dot_circles_unachieved_targets(self,pos,env):
        column, row = pos
        for agent in env.agent_list:
            for i in range(len(agent.targets)):
                if pos == agent.targets[i] and i == 0 and i >= agent.mode:
                    self.screen.blit(self.c_[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                if pos == agent.targets[i] and i == 1 and i >= agent.mode:
                    self.screen.blit(self.o_[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                if pos == agent.targets[i] and i == 2 and i >= agent.mode:
                    self.screen.blit(self.flag,
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                
    # this function puts the targets of the minecraft mode of visualization on the display
    def put_targets_minecraft(self,pos,env):
        column, row = pos
        for agent in env.agent_list:
            for i in range(len(agent.targets)):
                if pos == agent.targets[i]:
                    self.screen.blit(self.h_[agent.agent_id],
                                        ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))

    # this function puts the unachieved targets of the minecraft mode of visualization on the display
    def put_targets_minecraft_unachieved_targets(self,pos,env):
        column, row = pos
        for agent in env.agent_list:
            for i in range(len(agent.targets)):
                if pos == agent.targets[i] and i>= agent.mode :
                    self.screen.blit(self.h_[agent.agent_id],
                                        ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))

    #this method puts the agent image for the robot mode of visualization on the screen
    def put_agent_block(self,pos,env):
        column, row = pos
        for agent in env.agent_list:
            if pos == agent.location:
                self.screen.blit(self.img_robots[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
        return False
    
    #this method puts the agent image for the minecraft mode of visualization on the screen
    def put_minecraft_agents(self,pos,env):
        column, row = pos
        for agent in env.agent_list:

            if pos == agent.location and agent.location == agent.initial_location and agent.mode == 0:
                self.screen.blit(self.mc_[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
            elif pos == agent.location and agent.location == agent.initial_location and agent.mode == 1:
                self.screen.blit(self.mc1h_[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
            elif pos == agent.location and agent.location == agent.initial_location and agent.mode == 2:
                self.screen.blit(self.mc2h_[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
            elif pos == agent.location and agent.location == agent.initial_location and agent.mode == 3:
                self.screen.blit(self.mc3h_[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))


            #---
            if pos == agent.location  and  agent.mode == 0:
                self.screen.blit(self.m_[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
            elif pos == agent.location and agent.mode == 1:
                self.screen.blit(self.m1h_[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
            elif pos == agent.location and agent.mode == 2:
                self.screen.blit(self.m2h_[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
            elif pos == agent.location and agent.mode == 3:
                self.screen.blit(self.m3h_[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
            
    #this method puts the agent image for the dots-circles mode of visualization on the screen
    def put_dots_as_agents(self,pos,env):
        column, row = pos
        for agent in env.agent_list:
            
            if pos == agent.location and agent.mode == 0:
                self.screen.blit(self.d_[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
            elif pos == agent.location and agent.mode == 1:
                self.screen.blit(self.dc_[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
            elif pos == agent.location and agent.mode == 2:
                self.screen.blit(self.dco_[agent.agent_id],
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
            
            
    # this function renders the current state of the environment and everything in it.
    def render(self, env,step):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        pygame.display.update()
        self.clock.tick(600)
        pygame.display.flip()
        self.screen.fill(BLACK)
        text = self.my_font.render(f"Step: {step}", 1, WHITE)
        self.screen.blit(text, (5, 15))

        if self.representation_mode == 0:
            self.robots_render(env)
        elif self.representation_mode==1:
            self.dot_and_circles_render(env)
        elif self.representation_mode==2:
            self.minecraft_render(env)
                    
        pygame.display.update()

    #renderers

    #this is a helper function to render the minecraft mode of visualization
    def minecraft_render(self,env):
        for row in range(self.grid_size):
            for column in range(self.grid_size):
                pos = (column,row)
                pygame.draw.rect(self.screen, WHITE,
                                    [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50, WIDTH,
                                     HEIGHT])
                
                if pos in env.wall_blocks:
                    self.screen.blit(self.img_wall,
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))

                self.put_targets_minecraft_unachieved_targets(pos,env)

                self.put_minecraft_agents(pos,env)

    #this is a helper function to render the dots-circles mode of visualization
    def dot_and_circles_render(self,env):
        for row in range(self.grid_size):
            for column in range(self.grid_size):
                pos = (column,row)
                pygame.draw.rect(self.screen, WHITE,
                                    [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50, WIDTH,
                                     HEIGHT])
                
                if pos in env.wall_blocks:
                    self.screen.blit(self.img_wall,
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))

                #self.put_targets_dot_circles(pos,env)
                self.put_targets_dot_circles_unachieved_targets(pos,env)

                self.put_dots_as_agents(pos,env)

    #this is a helper function to render the robots mode of visualization
    def robots_render(self,env):
        for row in range(self.grid_size):
            for column in range(self.grid_size):
                pos = (column,row)
                pygame.draw.rect(self.screen, WHITE,
                                    [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50, WIDTH,
                                     HEIGHT])
                
                if pos in env.wall_blocks:
                    self.screen.blit(self.img_wall,
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
        
                elif self.is_target_block_v2(pos,env):
                    self.screen.blit(self.img_target,
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                
                self.put_agent_block(pos,env)

    # this method updates the display window and checks whether the
    #  user closed the window
    def keep(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        pygame.display.update()
        self.clock.tick(60)