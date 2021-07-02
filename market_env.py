from random import random
import numpy as np
import math

import gym
from gym import spaces

gym.logger.set_level(40)

import glob
import numpy as np
import pandas as pd
import os
import shutil 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from natsort import natsorted
#matplotlib inline

img_files = natsorted(glob.glob('img\\*')) 
files = [fn for fn in img_files]
#print(img_files)
#print(files)

#print(files)
IMG_WIDTH=224
IMG_HEIGHT=224
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)

train_files = [*img_files]
train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM, color_mode = "grayscale")) for img in train_files]
#train_imgs = [img_to_array(load_img(img)) for img in train_files]
train_imgs = np.array(train_imgs)
train_imgs = train_imgs.reshape(2981, 224, 224, 1)


class MarketEnv(gym.Env):

    PENALTY = 1 #0.999756079

    def __init__(self, dir_path, target_codes, input_codes, start_date, end_date, scope = 5, sudden_death = -1., cumulative_reward = False):
        self.action_space = spaces.Box(np.float32(np.array([3.0,3.5])), np.float32(np.array([4.0,4.5])))
        self.startDate = start_date
        self.endDate = end_date
        self.scope = scope
        self.sudden_death = sudden_death
        self.cumulative_reward = cumulative_reward
        
        #print(self.startDate)
        #print(self.endDate)
        
        self.inputCodes = []
        self.targetCodes = []
        self.dataMap = {}

        for code in (target_codes + input_codes):
            fn = dir_path + "./" + code + ".csv"

            data = {}
            lastClose = 0
            lastVolume = 0
            try:
                f = open(fn, "r")
                #print(f)
                for line in f:
                    if line.strip() != "":
                        dt, close = line.strip().split(",")
                        #print(dt)
                        #print(line.strip().split(","))
                        #print(line.strip().split(","))
                        #print(dt >= start_date)
                        try:
                            if dt >= start_date:
                                close = float(close)

                                if lastClose != 0 and close != 0:
                                    close_ = (close - lastClose) / lastClose

                                    data[dt] = (close_)
                                    #print(close_)
                                    #print(dt)
                                    #print(data.head(5))

                                lastClose = close
                        except(Exception, e):
                            print(e), line.strip().split(",")
                f.close()
            except(Exception, e):
                print(e)

            if len(data.keys()) > scope:
                #print(code)
                self.dataMap[code] = data
                #print(dataMap[code])
                if code in target_codes:
                    self.targetCodes.append(code)
                if code in input_codes:
                    self.inputCodes.append(code)

        self.actions = [
            "LONG",
            "SHORT",
        ]

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(np.ones(scope * (len(input_codes) + 1)) * -1, np.ones(scope * (len(input_codes) + 1)))
        self._reset()
        self._seed()

    def _step(self, action):
        if self.done:
            return self.state, self.reward, self.done, {}

        self.reward = 0
        if self.actions[action] == "LONG":
            if sum(self.boughts) < 0:
                for b in self.boughts:
                    self.reward += -(b + 1)
                    #print(b)
                    #print(self.boughts)
                if self.cumulative_reward:
                    self.reward = self.reward / max(1, len(self.boughts))

                if self.sudden_death * len(self.boughts) > self.reward:
                    self.done = True

                self.boughts = []

            self.boughts.append(1.0)
        elif self.actions[action] == "SHORT":
            if sum(self.boughts) > 0:
                for b in self.boughts:
                    self.reward += b - 1
                    #print(b)
                    #print(self.boughts)
                if self.cumulative_reward:
                    self.reward = self.reward / max(1, len(self.boughts))

                if self.sudden_death * len(self.boughts) > self.reward:
                    self.done = True

                self.boughts = []

            self.boughts.append(-1.0)
        else:
            pass

        vari = self.target[self.targetDates[self.currentTargetIndex]]
        #vari = self.target[self.targetDates[self.currentTargetIndex]][0]
        #print(vari)
        #print(int(random() * len(self.targetCodes)))
        self.cum = self.cum * (1 + vari)

        for i in range(len(self.boughts)):
            self.boughts[i] = self.boughts[i] * MarketEnv.PENALTY * (1 + vari * (-1 if sum(self.boughts) < 0 else 1))

        self.defineState()
        self.currentTargetIndex += 1
        if self.currentTargetIndex >= len(self.targetDates) or self.endDate <= self.targetDates[self.currentTargetIndex]:
            self.done = True

        if self.done:
            for b in self.boughts:
                self.reward += (b * (1 if sum(self.boughts) > 0 else -1)) - 1
        if self.cumulative_reward:
                self.reward = self.reward / max(1, len(self.boughts))

                self.boughts = []

        return self.state, self.reward, self.done, {"dt": self.targetDates[self.currentTargetIndex], "cum": self.cum, "code": self.targetCode}

    def _reset(self):
        
        self.targetCode = self.targetCodes[int(random() * len(self.targetCodes))]
        #print(self.targetCode)
        #self.targetCode = self.targetCodes[int(0)]
        #self.targetCode = '001111'
        self.target = self.dataMap[self.targetCode]
        #print(self.dataMap)
        #print(self.target)
        self.targetDates = sorted(self.target.keys())
        #print(sorted(self.target.keys()))
        self.currentTargetIndex = self.scope
        self.boughts = []
        self.cum = 1.
        #print(self.targetDates[self.currentTargetIndex])
        #print(self.targetDates)
        #print(self.currentTargetIndex)
        
        self.done = False
        self.reward = 0

        self.defineState()

        return self.state

    def _render(self, mode='human', close=False):
        if close:
            return
        return self.state

    '''
    def _close(self):
        pass

    def _configure(self):
        pass
    '''

    def _seed(self):
        return int(random() * 100)

    def defineState(self):
        tmpState = []

        #budget = (sum(self.boughts) / len(self.boughts)) if len(self.boughts) > 0 else 1.
        #size = math.log(max(1., len(self.boughts)), 100)
        #position = 1. if sum(self.boughts) > 0 else 0.
        #tmpState.append([[budget, size, position]])

        subject = []
        #subjectVolume = []
        for i in range(self.scope):
            #print("hello ", self.target[self.targetDates[self.currentTargetIndex - 1 - i]])
            try:
                #subject.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]]])
                subject.append([train_imgs[self.currentTargetIndex - 1 - i]])
                #subjectVolume.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][3]])
            except(Exception, e):
                print(self.targetCode, self.currentTargetIndex, i, len(self.targetDates))
                self.done = True
        #tmpState.append(subject)

        #tmpState = [np.array(i) for i in tmpState]
        tmpState = [np.array(i) for i in subject]
        self.state = tmpState

