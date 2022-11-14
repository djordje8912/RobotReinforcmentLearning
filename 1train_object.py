#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 17:55:43 2021

@author: hal9000
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 05:02:53 2021

@author: hal9000
"""

import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import socket
import gym
# import roboschool
import random

# import gripper_env as grip

# import pybullet_envs

from ppo import PPO

import math
import json
# from numba import jit,cuda

vacuum=False

def linija(x1,y1,z1,x2,y2,z2,conn):
    offsetx=int((x2-x1)/0.02)
    offsety=int((y2-y1)/0.02)
    offsetz=int((z2-z1)/0.02)
    print('koraka po x-u: {} koraka po x-u: {}  koraka po x-u: {}'.format(offsetx,offsety,offsetz))
    komande=[]
    if(offsetx>0):
        for i in range(0,offsetx):
            komande.append(0)
          
    else:
        for i in range(0,-offsetx):
            komande.append(1)
           

    
    if(offsety>0):
        for i in range(0,offsety):
            komande.append(2)
          
    else:
        for i in range(0,-offsety):
            komande.append(3)
           

    if(offsetz>0):
        for i in range(0,offsetz):
            komande.append(4)
           
    else:
        for i in range(0,-offsetz):
            komande.append(5)
                     
    for k in komande:
         izvrsi_komandu(k,conn)
def izvrsi_komandu2(komanda,conn):
    global vacuum
    # komanda=input("Unesi komandu-->")
    # print(ord(komanda))
   # try:  # used try so that if user pressed other than the given key error will not be shown
    if komanda==0:  # if key 'q' is pressed 
        print('komanda x++ !')   #49 ascii
        # conn.send(str(1).encode('utf-8'))
        conn2.send(str(0).encode('utf-8'))
        #break  # finishing the loop
    elif komanda==1:  # if key 'q' is pressed 
        print('komanda x-- !')
        # conn.send(str(2).encode('utf-8'))
        conn2.sendall(str(1).encode('utf-8'))
        #break  # finishing the loop
    elif komanda==2:  # if key 'q' is pressed 
        print('komanda y++ !')
        # conn.send(str(3).encode('utf-8'))
        conn2.sendall(str(2).encode('utf-8'))
        #break  # finishing the loop
    elif komanda==3:  # if key 'q' is pressed 
        print('komanda y-- !')
        # conn.send(str(4).encode('utf-8'))
        conn2.sendall(str(3).encode('utf-8'))
        #break  # finishing the loop11
    elif komanda==4:  # if key 'q' is pressed 
        print('komanda z++ !')
        # conn.send(str(5).encode('utf-8'))
        conn2.sendall(str(4).encode('utf-8'))
        #break  # finishing the loop
    elif komanda==5:  # if key 'q' is pressed 
        print('komanda z-- !')
        # conn.send(str(6).encode('utf-8'))
        conn2.sendall(str(5).encode('utf-8'))
        #break  # finishing the loop
    elif komanda==ord('7'):  # if key 'q' is pressed 
        print('komanda r++ !')
        # conn.send(str(7).encode('utf-8'))
        conn2.sendall(str(6).encode('utf-8'))
        #break  # finishing the loop
    elif komanda==ord('8'):  # if key 'q' is pressed 
        print('komanda r-- !')
        # conn.send(str(8).encode('utf-8'))
        conn2.sendall(str(7).encode('utf-8'))
        #break  # finishing the loop
    elif komanda==ord('/'):  # if key 'q' is pressed 
        # print('komanda max pos !') #48 ascii
        conn2.sendall(str('/').encode('utf-8'))
        #break  # finishing the loop     
    elif komanda==ord('0'):  # if key 'q' is pressed 
        # print('komanda home !')  #47 ascii
       conn.send(str(0).encode('utf-8'))
    elif komanda==6 and (vacuum==True):  # if key 'q' is pressed 
        # print('komanda vakuum iskljuci !')
        conn2.send(str('+').encode('utf-8'))
        vacuum=False
        # conn2.sendall(str(8).encode('utf-8'))
    elif komanda==6 and (vacuum==False):  # if key 'q' is pressed 
        # print('komanda vakuum ukljuci !')
        conn2.send(str('+').encode('utf-8')) 
        vacuum=True
        # conn2.sendall(str(8).encode('utf-8'))
        #break  # finishing the loop                      
   # except:
    #    break  # if user pressed a key other than the given key the loop will breakqqqq
    
    data = conn.recv(1)
    if not data:
        return False
        # break
    # conn.send(data)
    return True 

def izvrsi_komandu(komanda,conn):
    global vacuum
    # komanda=input("Unesi komandu-->")
    # print(ord(komanda))
   # try:  # used try so that if user pressed other than the given key error will not be shown
    if komanda==0:  # if key 'q' is pressed 
        print('komanda x++ !')   #49 ascii
        conn.send(str(2).encode('utf-8'))
        # conn2.send(str(0).encode('utf-8'))
        #break  # finishing the loop
    elif komanda==1:  # if key 'q' is pressed 
        print('komanda x-- !')
        conn.send(str(1).encode('utf-8'))
        # conn2.sendall(str(1).encode('utf-8'))
        #break  # finishing the loop
    elif komanda==2:  # if key 'q' is pressed 
        print('komanda y++ !')
        conn.send(str(4).encode('utf-8'))
        # conn2.sendall(str(2).encode('utf-8'))
        #break  # finishing the loop
    elif komanda==3:  # if key 'q' is pressed 
        print('komanda y-- !')
        conn.send(str(3).encode('utf-8'))
        # conn2.sendall(str(3).encode('utf-8'))
        #break  # finishing the loop11
    elif komanda==4:  # if key 'q' is pressed 
        print('komanda z++ !')
        conn.send(str(5).encode('utf-8'))
        # conn2.sendall(str(4).encode('utf-8'))
        #break  # finishing the loop
    elif komanda==5:  # if key 'q' is pressed 
        print('komanda z-- !')
        conn.send(str(6).encode('utf-8'))
        # conn2.sendall(str(5).encode('utf-8'))
        #break  # finishing the loop
    elif komanda==7:  # if key 'q' is pressed 
        print('komanda r++ !')
        conn.send(str(7).encode('utf-8'))
        # conn2.sendall(str(6).encode('utf-8'))
        #break  # finishing the loop
    elif komanda==8:  # if key 'q' is pressed 
        print('komanda r-- !')
        conn.send(str(8).encode('utf-8'))
        # conn2.sendall(str(7).encode('utf-8'))
        #break  # finishing the loop
    elif komanda==ord('/'):  # if key 'q' is pressed 
        print('komanda max pos !') #48 ascii
        conn.sendall(str('/').encode('utf-8'))
        #break  # finishing the loop     
    elif komanda==ord('0'):  # if key 'q' is pressed 
        print('komanda home !')  #47 ascii
        conn.send(str('0').encode('utf-8'))
    elif komanda==6 and (vacuum==True):  # if key 'q' is pressed 
        print('komanda vakuum iskljuci !')
        conn.send(str('-').encode('utf-8'))
        vacuum=False
        # conn2.sendall(str(8).encode('utf-8'))
    elif komanda==6 and (vacuum==False):  # if key 'q' is pressed 
        print('komanda vakuum ukljuci !')
        conn.send(str('+').encode('utf-8')) 
        vacuum=True
        # conn2.sendall(str(8).encode('utf-8'))
        #break  # finishing the loop                      
   # except:
    #    break  # if user pressed a key other than the given key the loop will breakqqqq
    
    data = conn.recv(1)
    if not data:
        return False
        # break
    # conn.send(data)
    return True 



class env():
    botX=0
    botY=0
    botZ=0
    grip_offsetH=0.03
    grip_offsetS=0.1
    objX=0
    objY=0
    objZ=0.02
    timestamp=0
    episodes=1

    objtargetX=0
    objtargetY=0
    objtargetZ=0.02

    objectHP=0.02
    objectSP=0.12
    
    first_vacuum=False

    offset=objectSP-grip_offsetS
    step_translate=0.02
    object_uhvacen=False
    vacuum_on=False

    first=True

    position_award=0
    position_award_previous=0
    def __init__(self):
        self.objtargetX=round(random.uniform(-0.2,0.2),2)
        self.objtargetY=round(random.uniform(-0.2,0.2),2)
        self.objtargetZ=0.02

        self.objX=round(random.uniform(-0.2,0.2),2)
        self.objY=round(random.uniform(-0.2,0.2),2)
        self.objZ=0.02

        self.botX=round(random.uniform(-0.2,0.2),2)
        self.botY=round(random.uniform(-0.2,0.2),2)
        self.botZ=round(random.uniform(0.05,0.2),2)

    def get_award_value(self):
        return self.position_award
    def pos_award(self):
        #print('racunanje nagrade ')
        #if(self.first):
        #    self.position_award_previous=1/(0.01+math.sqrt((self.objX-self.objtargetX)**2+(self.objY-self.objtargetY)**2))/100
        #    self.first=False
        #    return 0

        if(self.object_uhvacen):
            self.position_award=math.sqrt((self.objX-self.objtargetX)**2+(self.objY-self.objtargetY)**2+(self.objZ-self.objtargetZ)**2)
        else:
            self.position_award=math.sqrt((self.objX-self.botX)**2+(self.objY-self.botY)**2+(self.objZ-self.botZ)**2)
        award=self.position_award-self.position_award_previous
        # print(award,self.position_award, self.position_award_previous)
        self.position_award_previous=self.position_award

        
        if(math.sqrt((self.objX-self.objtargetX)**2+(self.objY-self.objtargetY)**2)<=0.03):
            return 1
        if(award<0):
            return 0.05
        else:
            return -0.05


    def gripperAboveObject(self):
      
        if(abs(self.objX-self.botX) <self.offset and   abs(self.objY-self.botY) <self.offset ):
            return True

        return False
    def intersection(self,a,b,c,d,m):
        am=a-m
        ab=a-b
        ad=a-d

        return (np.dot(am,ab)>0 and np.dot(am,ab)<np.dot(ab,ab) and np.dot(am,ad)>0 and np.dot(am,ad)<np.dot(ad,ad))
             
    def poklapanje(self):
        m1=np.array([self.objX,self.objY])
        a1=np.array([self.objX-self.objectSP,self.objY-self.objectSP])
        b1=np.array([self.objX-self.objectSP,self.objY+self.objectSP])
        c1=np.array([self.objX+self.objectSP,self.objY+self.objectSP])
        d1=np.array([self.objX+self.objectSP,self.objY-self.objectSP])
      
        m2=np.array([self.botX,self.botY])
        a2=np.array([self.botX-self.grip_offsetS,self.botY-self.grip_offsetS])
        b2=np.array([self.botX-self.grip_offsetS,self.botY+self.grip_offsetS])
        c2=np.array([self.botX+self.grip_offsetS,self.botY+self.grip_offsetS])
        d2=np.array([self.botX+self.grip_offsetS,self.botY-self.grip_offsetS])

        if(self.intersection(a1,b1,c1,d1,m2)):
            return True
        if(self.intersection(a1,b1,c1,d1,a2)):
            return True
        if(self.intersection(a1,b1,c1,d1,b2)):
            return True
        if(self.intersection(a1,b1,c1,d1,c2)):
            return True
        if(self.intersection(a1,b1,c1,d1,d2)):
            return True
        if(self.intersection(a2,b2,c2,d2,m1)):
            return True
        if(self.intersection(a2,b2,c2,d2,a1)):
            return True
        if(self.intersection(a2,b2,c2,d2,b1)):
            return True
        if(self.intersection(a2,b2,c2,d2,c1)):
            return True
        if(self.intersection(a2,b2,c2,d2,d1)):
            return True

        return False
        


    def vacuum_logic(self):
        self.vacuum_on=not self.vacuum_on
        if(self.object_uhvacen==True):
            self.object_uhvacen=False
            objZ=0.02
        else:
            if(self.botZ>=self.objZ+self.objectHP and self.botZ<=self.objZ+self.objectHP+self.grip_offsetH and self.gripperAboveObject()):
                self.object_uhvacen=True
                # print(self.object_uhvacen)
                return True
        return False
            
    def move(self,action):
        movX=0
        movY=0
        movZ=0
        if(action==0):
            movX+=self.step_translate
        if(action==1):
            movX-=self.step_translate
        if(action==2):
            movY+=self.step_translate
        if(action==3):
            movY-=self.step_translate
        if(action==4):
            movZ+=self.step_translate
        if(action==5):
            movZ-=self.step_translate
        self.botX+=movX
        self.botY+=movY
        self.botZ+=movZ
        if(self.object_uhvacen==True):
            self.objX+=movX
            self.objY+=movY
            self.objZ+=movZ
            # print('move object')
            
            
    def reset(self):
        self.episodes+=1
        self.timestamp=0
        self.vacuum_on=False
        self.object_uhvacen=False
        self.first=True
        
        self.first_vacuum=False
        self.position_award=0
        self.position_award_previous=math.sqrt((self.objX-self.botX)**2+(self.objY-self.botY)**2+(self.objZ-self.botZ)**2)
        
        # k=min(0.15,0.05*int(math.log(self.episodes,20)))
        k=0.5
        # print(k)
        self.objtargetX=round(random.uniform(-k,k),2)
        self.objtargetY=round(random.uniform(-k,k),2)
        self.objtargetZ=0.02

        self.objX=round(random.uniform(-k,k),2)
        
        self.objY=round(random.uniform(-k,k),2)
        while(abs(self.objY-self.objtargetY)<0.02):
            self.objY=round(random.uniform(-k,k),2)
        self.objZ=0.02

        self.botX=round(random.uniform(-k,k),2)
        self.botY=round(random.uniform(-k,k),2)
        self.botZ=round(random.uniform(0.05,max(k,0.05)),2)
        
        # print('bot=[{},{},{}]   obj=[{},{},{}] '.format(self.botX,self.botY,self.botZ,self.objX,self.objY,self.objZ))
            
        return np.array( [self.botX-self.objX,self.botY-self.objY,self.botZ-self.objZ,self.botX-self.objtargetX,self.botY-self.objtargetY,self.botZ-self.objtargetZ,self.vacuum_on,self.object_uhvacen])
    def step(self,action):
         self.timestamp+=1
        
         
         if(self.timestamp==149):
             return  np.array([self.botX-self.objX,self.botY-self.objY,self.botZ-self.objZ,self.botX-self.objtargetX,self.botY-self.objtargetY,self.botZ-self.objtargetZ,self.vacuum_on,self.object_uhvacen]),-1,True

         if(action!=6):
            self.move(action)
         if(action==6):
            vacuum=self.vacuum_logic()
            if(vacuum):
                if(self.first_vacuum==False):
                    self.first_vacuum=True
                    return  np.array([self.botX-self.objX,self.botY-self.objY,self.botZ-self.objZ,self.botX-self.objtargetX,self.botY-self.objtargetY,self.botZ-self.objtargetZ,self.vacuum_on,self.object_uhvacen]),0.8,False
                    
                else:
                   return  np.array([self.botX-self.objX,self.botY-self.objY,self.botZ-self.objZ,self.botX-self.objtargetX,self.botY-self.objtargetY,self.botZ-self.objtargetZ,self.vacuum_on,self.object_uhvacen]),-0.1,False
                     
         if self.collision_detect():
            return  np.array([self.botX-self.objX,self.botY-self.objY,self.botZ-self.objZ,self.botX-self.objtargetX,self.botY-self.objtargetY,self.botZ-self.objtargetZ,self.vacuum_on,self.object_uhvacen]),-0.8,True
        
         aw=self.pos_award()
         if(aw==1):
            return  np.array([self.botX-self.objX,self.botY-self.objY,self.botZ-self.objZ,self.botX-self.objtargetX,self.botY-self.objtargetY,self.botZ-self.objtargetZ,self.vacuum_on,self.object_uhvacen]),1,True
         else:
            return  np.array([self.botX-self.objX,self.botY-self.objY,self.botZ-self.objZ,self.botX-self.objtargetX,self.botY-self.objtargetY,self.botZ-self.objtargetZ,self.vacuum_on,self.object_uhvacen]),aw,False


    def observation(self):
        return [self.botX-self.objX,self.botY-self.objY,self.botZ-self.objZ,self.botX-self.objtargetX,self.botY-self.objtargetY,self.botZ-self.objtargetZ,self.vacuum_on,self.object_uhvacen]
    def collision_detect(self):
        sudar=False
        #udaranje grippera u ravan
        if(self.botZ<0):
            print('udaranje grippera u ravan')
            return True

        

        #udaranje objekta u ravan
        if(self.objZ-self.objectHP<0):
            print('udaranje objekta u ravan')
            return True

        #udaranje grippera u objekat
        if(self.botZ<self.objZ+self.objectHP):
            if(self.poklapanje()):
                print('udaranje grippera u objekat')
                return True

        return False





################################### Training ###################################

def train():

    print("============================================================================================")


    ####### initialize environment hyperparameters ######

    env_name = "Vacuum gripper"

    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 200                   # max timesteps in one episode
    max_training_episodes = 100000  # break training loop if timeteps > max_training_timesteps

    print_freq = 50     # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

    #####################################################


    ## Note : print/log frequencies should be > than max_ep_len


    ################ PPO hyperparameters ################

    update_timestep = max_ep_len * 100    # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.001     # learning rate for actor network 0.0003
    lr_critic = 0.002      # learning rate for critic network 0.001 

    random_seed = 0         # set random seed if required (0 = no random seed)

    #####################################################



    print("training environment name : " + env_name)
    
    en = env()
    en.reset()
    # state space dimension
    # o = env.observation()
    
    state_dim=8

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = 7



    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten

    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)


    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)


    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    #####################################################


    ################### checkpointing ###################

    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)

    #####################################################


    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")

    print("max training episodes : ", max_training_episodes)
    print("max timesteps per episode : ", max_ep_len)

    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

    print("--------------------------------------------------------------------------------------------")

    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")

    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")

    else:
        print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")

    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")

    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # ppo_agent.load( './PPO_preTrained/Vacuum gripper/PPO_Vacuum gripper_0_0.pth') #novi input 32,64
    # ppo_agent.load( './PPO_preTrained/Vacuum gripper/PPO_Vacuum PPO_Vacuum gripper_0_0-0.7.pth') #novi input 32,64
     
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")


    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')


    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    last_eps_time=0
    
    average_max_award=0
    
    test_set=250
    
    solved=np.zeros(test_set)
    passed_test=np.ones(test_set)
    max_episode_award=0
    reward=0
    timestep_diff=0
    timestep2=0
    # training loop
    while i_episode <= max_training_episodes:
        # print('Last reward : ',reward)   
        if reward > 0  :
            # print(reward,i_episode)
            solved[i_episode%len(solved)]+=1
            
            # print(str(solved))
            print('solved')
            # print(str(i_episode))
            if(np.all(np.greater(solved,passed_test))):
                print("########## Solved! ##########")
                ppo_agent.save('./{}_solved.pth'.format(env_name))
                break
               
        else:
            solved=np.zeros(250)
            # printing average reward
        
        state = en.reset()
      
        # current_ep_reward = 0
        # print('----{}----'.format(i_episode% print_freq))
        if i_episode % print_freq== 0 : 
               
               # print average reward till last episode
               # print_avg_reward = print_running_reward / print_running_episodes
               # print_avg_reward = round(print_avg_reward, 2)
               average_max_award=average_max_award/print_freq
               timestep_diff=time_step-timestep2
               timestep2=time_step
               print("Episode : {} \t\t Average Timestep : {} \t\t Average Reward : {}".format(i_episode, timestep_diff/print_freq, round(average_max_award,3)))
               average_max_award=0
               max_episode_award=0
               # print_running_reward = 0
               # print_running_episodes = 0
                
        average_max_award+=max_episode_award
        max_episode_award=0
        for t in range(1, max_ep_len+1):
            
            
            # if(t==max_ep_len):
                
            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done = en.step(action)
            # print(' [{}] '.format(action+1),end='')
            # reward*=100
            
            
            # print('state:{}  action:{} reward:{}'.format(state,action,reward))
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward+0*(sum(solved)/len(solved)))
            ppo_agent.buffer.is_terminals.append(done)
            
            
            
            # ppo_agent.buffer.states.append(state)
         

            time_step +=1
            # current_ep_reward += reward
            
            # max_episode_award=max(max_episode_award,reward)
            max_episode_award+=reward
            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            # if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            #     ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            # if time_step % log_freq == 0:

            #     # log average reward till last episode
            #     log_avg_reward = log_running_reward / log_running_episodes
            #     log_avg_reward = round(log_avg_reward, 4)

            #     log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            #     log_f.flush()

            #     log_running_reward = 0
            #     log_running_episodes = 0

            
           
           
                # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
            
                break

        # print_running_reward += current_ep_reward
        print_running_episodes += 1

        # log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
        


    log_f.close()
   




    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")




if __name__ == '__main__':
    rezimi=['man_python_env','ppo_python_treniranje','man_unity','automatski','man_zadatak','automatski_proceduralno']
    rezim_rada=rezimi[3]
    
    if(rezim_rada=='automatski_proceduralno'):
        HOST = '192.168.0.6'  # Standard loopback interface address (localhost)
        PORT = 5001 
        s=socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
         
        print('Connected with robot', addr)
        time.sleep(3)
        comm_ok=True
        print('inicijalizacija....')
        botx=0.0
        boty=0.0
        # en.botZ=0.12
        botz=0.2
         
        read=False
        while( read==False):
            time.sleep(2)
            print('citanje...')
            try:    
                    
            # Opening JSON file
                f = open('request/request.json',)
                read=True 
            except:
                pass    
          # returns JSON object as 
          # a dictionary
        data = json.load(f)
        
          # Iterating through the json
          # list
        for i in data:
            print(i)
        
          # Closing file
        f.close()
        os.remove("request/request.json")
                 
                  
        objx=float(data['startX'])
        objy=float(data['startY'])
        objz=0.05
         
        targetx=float(data['targetX'])
        targety=float(data['targetY'])
        targetz=0.05
        
        print('objx:{} objy:{} objz:{}  tx={} ty={} tz={}'.format(objx,objy,objz,targetx,targety,targetz))
        izvrsi_komandu(48,conn)
         
        linija(botx,boty,botz,objx,objy,objz+0.1,conn)
        linija(objx,objy,objz+0.1,objx,objy,objz,conn)
        izvrsi_komandu(6,conn)
        linija(objx,objy,objz,objx,objy,objz+0.1,conn)
        linija(objx,objy,objz+0.1,targetx,targety,targetz+0.1,conn)
        linija(targetx,targety,targetz+0.1,targetx,targety,targetz,conn)
        izvrsi_komandu(6,conn)
        izvrsi_komandu(4,conn)
        izvrsi_komandu(4,conn)
        izvrsi_komandu(4,conn)
        izvrsi_komandu(47,conn)
    if(rezim_rada=='automatski'):
        HOST = '192.168.0.6'  # Standard loopback interface address (localhost)
        PORT = 5001 
        s=socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
            
        print('Connected with robot', addr)
        time.sleep(3)
        comm_ok=True
        print('inicijalizacija....')

        ppo_agent = PPO(8, 7, 0.0006, 0.001 , 0.99, 80, 0.2  , False, 0.6)
        ppo_agent.load( './PPO_preTrained/Vacuum gripper/PPO_Vacuum gripper_0_0.pth')
        en=env()
        done =False
        en.botX=0.0
        en.botY=0.0
        # en.botZ=0.12
        en.botZ=0.2
        
        # en.objX=0.0
        # en.objY=0.00
        # en.objZ=0.02
        
        # en.objtargetX=0.2
        # en.objtargetY=0.3
        # en.objtargetZ=0.02
        
        while(True):
            read=False
            while( read==False):
                time.sleep(2)
                print('citanje...')
                try:    
                    
                # Opening JSON file
                    f = open('request/request.json',)
                    read=True 
                except:
                    pass    
            # returns JSON object as 
            # a dictionary
            data = json.load(f)
              
            # Iterating through the json
            # list
            for i in data:
                print(i)
              
            # Closing file
            f.close()
            os.remove("request/request.json")
           
            
            en.objX=float(data['startX'])
            en.objY=float(data['startY'])
            
            
            en.objtargetX=float(data['targetX'])
            en.objtargetY=float(data['targetY'])
           
            
            # en.vacuum_gripper.change_position(0.3,0.0,0.08)
            # en.add_telo_m(-0.2,0.00,0.02)  
            # en.add_cilj_m(0.09,0.00,0.02)
            
            
            state=en.observation()
            best_action=[]
            curr_action=[]
            curr_award=0
            best_award=0
            for i in (1,10):
                while(done!= True):
                    
                    
                    
                    # print(data)
                    # comm_ok=izvrsi_komandu('0',conn,conn2)
                    # print(comm_ok)
                    # time.sleep(1)
                    # comm_ok=izvrsi_komandu('0',conn,conn2)
                    print('a')
                    
                    action = ppo_agent.select_action(state)
                    
                    state, reward, done = en.step(action)
                    curr_action.append(action)
                    curr_award=reward
                if(curr_award>best_award):
                    best_award=reward
                    best_action=curr_action
                    curr_state=[]
            print(' action:{} best reward:{}'.format(best_action,best_award)) 
            
            
            
            izvrsi_komandu(48,conn)
            for i in best_action:
               
                # comm_ok=izvrsi_komandu('0',conn,conn2)
                # print(comm_ok)
                # time.sleep(1)
                # comm_ok=izvrsi_komandu('0',conn,conn2)
                print(comm_ok)
                
                izvrsi_komandu(i,conn)
                print(i)
            
           
            izvrsi_komandu(6,conn)
           
           
            izvrsi_komandu(4,conn)
            izvrsi_komandu(4,conn)
            izvrsi_komandu(47,conn)
    if(rezim_rada=='man_zadatak'):
        ppo_agent = PPO(8, 7, 0.0006, 0.001 , 0.99, 80, 0.2  , False, 0.6)
        ppo_agent.load( './PPO_preTrained/Vacuum gripper/PPO_Vacuum gripper_0_0.pth')
        en=env()
        done =False
        en.botX=0.0
        en.botY=0.0
        en.botZ=0.2
        
        en.objX=0.1
        en.objY=0.2
        en.objZ=0.02
        
        en.objtargetX=0.2
        en.objtargetY=0.3
        en.objtargetZ=0.02
        
        # en.vacuum_gripper.change_position(0.3,0.0,0.08)
        # en.add_telo_m(-0.2,0.00,0.02)  
        # en.add_cilj_m(0.09,0.00,0.02)
        
        
        state=en.observation()
        best_action=[]
        curr_action=[]
        curr_award=0
        best_award=0
        for i in (1,1):
            while(done!= True):
                
                
                
                # print(data)
                # comm_ok=izvrsi_komandu('0',conn,conn2)
                # print(comm_ok)
                # time.sleep(1)
                # comm_ok=izvrsi_komandu('0',conn,conn2)
                print('a')
                
                action = ppo_agent.select_action(state)
                
                state, reward, done = en.step(action)
                curr_action.append(action)
                curr_award=reward
            if(curr_award>best_award):
                best_award=reward
                best_action=curr_action
                curr_state=[]
        print(' action:{} best reward:{}'.format(best_action,best_award))    
    if(rezim_rada=='man_python_env'):
        en=env()
        print(en.observation())
        while(True):
            komanda=int(input('Unesi komandu: '))
            state,reward,done=en.step(komanda)
            print('stanje:{} nagrada:{} '.format(state,reward))
            if(done==True):
                state=en.reset()
                print('stanje:{} '.format(state,reward))
            
    if(rezim_rada=='ppo_python_treniranje'):
        train()
    if(rezim_rada=='man_unity'):
        
        HOST = '192.168.0.6'  # Standard loopback interface address (localhost)
        PORT = 5001        # Port to listen on (non-privileged ports are > 1023)
        vacuum=False
        MAN=True
        HOST2 = 'localhost'  
        PORT2 = 5554 
        
        ppo_agent = PPO(10, 7, 0.0006, 0.001 , 0.99, 80, 0.2  , False, 0.6)

        # ppo_agent.load( './PPO_preTrained/Vacuum gripper/PPO_Vacuum gripper_0_0.pth') #novi input 32,64
        # ppo_agent.load( './PPO_Vacuum gripper_0_0.pth') #novi input 32,64
        

        s2=socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        s2.bind((HOST2, PORT2))
        s2.listen()
        conn2, addr2 = s2.accept()
        print('Connected with Unity', addr2)
        # s=socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        # s.bind((HOST, PORT))
        # s.listen()
        # conn, addr = s.accept()
            
        # print('Connected with robot', addr)
        
        # time.sleep(3)
        comm_ok=True
        print('inicijalizacija....')
        
        en=env(False)
        done =False
        print('1')
        en.vacuum_gripper.change_position(0.0,0.0,0.08)
        print('2')
        en.add_telo_m(0.0,0.00,0.02)  
        print('3')
        en.add_cilj_m(0.1,0.00,0.02)
        print(en.ciljevi)
        state=en.observation()
        while(done!= True):
            
            
            
            # print(data)
            # comm_ok=izvrsi_komandu('0',conn,conn2)
            # print(comm_ok)
            # time.sleep(1)
            # comm_ok=izvrsi_komandu('0',conn,conn2)
            print(comm_ok)
            
            action = ppo_agent.select_action(state)
            print(action)
            state, reward, done = en.step(action+1)
            izvrsi_komandu2(action+1,conn2)
            
            # conn2.send(str(1).encode('utf-8'))
            
            
            
            # while True:
            #     pass
        