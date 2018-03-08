import ast
import json

import datetime
import threading
from time import sleep

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deterministic import myGen_DF,myGen_File
from A3C_DataCollector import Bitmex,BitmexTest,BitmexTestv2
from OrderManagerLive import OrderManagerLive
from DataCollectorLive import DataCollectorLive

class Micro_Arch_4():

    def __init__(self, INIT_AMOUNT, LEV, STOP_LOSS_PERCENTAGE,NORM_DATA_1, DATA_ACTUAL_PRICE, NORM_ACTUAL_PRICE_1,hist_Vals):
        self.LEVERAGE = LEV
        self.INIT_AMOUNT = INIT_AMOUNT  # 10
        self.STOP_LOSS_PERCENTAGE = STOP_LOSS_PERCENTAGE
        self.hist_Vals = hist_Vals
        self.savedActions = []
        self.obsAction = 0
        self.DATA_GEN_DF = NORM_DATA_1
        self.DATA_GEN = myGen_DF(df=NORM_DATA_1)
        self.DATA_GEN_ACTUAL_PRICE_DF = DATA_ACTUAL_PRICE
        self.DATA_GEN_ACTUAL_PRICE = myGen_DF(df=DATA_ACTUAL_PRICE)
        self.DATA_GEN_ACTUAL_PRICE_NORM_DF = NORM_ACTUAL_PRICE_1
        self.DATA_GEN_ACTUAL_PRICE_NORM = myGen_DF(df=NORM_ACTUAL_PRICE_1)
        #DATA_GEN
        #DATA_GEN_ACTUAL_PRICE  # actual bid,ask price from order book
        #DATA_GEN_ACTUAL_PRICE_NORM

        self.rollPrice = 0
        self.SLPrice = 0
        self.ExitPrice = 0
        self.runningPnL = 0

        self.INITIAL_PRICE = 0
        self.nbBuy = 0
        self.nbHold = 0
        self.nbSell = 0
        self.percentBias = 0.3
        return

    def reset(self):
        self.savedEntryActions = []
        self.obsAction = 0
        self.DATA = []
        self.DATA_PRICE = []
        self.DATA_PRICE_NORM = []
        self.DATA_GEN.rewind()
        self.DATA_GEN_ACTUAL_PRICE.rewind()
        self.DATA_GEN_ACTUAL_PRICE_NORM.rewind()
        self.runningPnL = 0
        self.INITIAL_PRICE = 0
        self.nbBuy = 0
        self.nbHold = 0
        self.nbSell = 0

        # GET FIRST PRICE DATA FOR OBSERVATION
        for i in range(self.hist_Vals):
            self.DATA.append(self.DATA_GEN.next())
            self.DATA_PRICE.append(self.DATA_GEN_ACTUAL_PRICE.next())
            self.DATA_PRICE_NORM.append(self.DATA_GEN_ACTUAL_PRICE_NORM.next())
        observation = self.get_obs()

        return observation

    def step(self, action):
        self.obsAction = action
        #print('action: ' + str(action))
        if action == 0:
            action = 'BUY'
        if action == 1:
            action = 'HOLD'
        if action == 2:
            action = 'SELL'

        done = False
        instant_pnl = 0.0
        reward = 0.0
        info = {}
        # print('action: ' + str(action))
        # reward = -self._time_fee
        if action == 'BUY':
            self.nbBuy = self.nbBuy + 1
            #tsBuy = self.DATA_GEN_DF.index.values[len(self.DATA_PRICE)-1]
            #print('tsBuy: ' + str(tsBuy))
            # count,bidP,askP,pcentPriceChgLong,pcentPriceChgShort,entryPriceLong,entryPriceShort,exitPriceLong,exitPriceShort,entryIndex,exitIndexLong,exitIndexShort
            #print('len(self.DATA_PRICE): ' + str(len(self.DATA_PRICE)))
            #print('BUY self.DATA_PRICE[-1][4]: ' + str(self.DATA_PRICE[-1][3]))

            percentPriceChg = self.DATA_PRICE[-1][3]-self.percentBias  #['bestBidPrice', 'bestAskPrice','pcentPriceChgLong','pcentPriceChgShort']
            self.DATA_PRICE[-1][3] = self.DATA_PRICE[-1][3] - self.percentBias
            #print('BUY percentPriceChg: ' + str(percentPriceChg))
            #print('BUY self.DATA_PRICE[-1][4]: ' + str(self.DATA_PRICE[-1][3]))

            self.savedEntryActions.append(['BUY',self.DATA_PRICE[-1]])
            reward = percentPriceChg

        if action == 'HOLD':
            self.nbHold = self.nbHold + 1
            self.savedEntryActions.append(['HOLD', self.DATA_PRICE[-1]])
        if action == 'SELL':
            self.nbSell = self.nbSell + 1
            #tsSell = self.DATA_GEN_DF.index.values[len(self.DATA_PRICE)-1]

            percentPriceChg = (-1 * self.DATA_PRICE[-1][4]) - self.percentBias  # ASK OFFER
            self.DATA_PRICE[-1][4] = (-1 * self.DATA_PRICE[-1][4]) - self.percentBias  # ASK OFFER

            self.savedEntryActions.append(['SELL', self.DATA_PRICE[-1]])
            reward = percentPriceChg

        instant_pnl = reward
        self.runningPnL = self.runningPnL + instant_pnl
        try:
            self.DATA.append(self.DATA_GEN.next())
            self.DATA_PRICE.append(self.DATA_GEN_ACTUAL_PRICE.next())
            self.DATA_PRICE_NORM.append(self.DATA_GEN_ACTUAL_PRICE_NORM.next())
        except StopIteration:
            done = True
            info['status'] = 'No more data.'

        observation = self.get_obs()
        # exit()
        return observation, reward, done, info

    def get_obs(self):

        #print('obs_Data: ' + str(len(self.DATA[-self.hist_Vals:][1:])))

        obs1 = np.asarray(self.DATA[-self.hist_Vals:], dtype=float)  # latest Data
        #obs2 = np.asarray([self.obsAction], dtype=float)
        #obs = np.concatenate([obs1, obs2])
        x,y = obs1.shape
        obs1 = obs1.reshape((1,(x*y)))
        #print()
        #print ('obs.shape ' +str(obs1.shape))
        return obs1

    def getEntryActionsLog(self):
        #print(str(self.savedEntryActions))
        return self.savedEntryActions

    def GET_DATA_GEN_ACTUAL_PRICE_DF(self):
        return self.DATA_GEN_ACTUAL_PRICE_DF
    def prntPerformance(self):

        ret = 'PnL: ' + str(self.runningPnL) + '\t'

        rewards = 'nbBuy : ' + str(self.nbBuy) + '\t' + \
                  'nbHold: ' + str(self.nbHold) + '\t' + \
                  'nbSell: ' + str(self.nbSell)
        print (ret)
        print (rewards)
        epPercentagePriceChg = self.runningPnL
        log = [epPercentagePriceChg, self.nbBuy, self.nbHold, self.nbSell]
        return log


class Micro_Arch_Live_Test():
    def __init__(self, INIT_AMOUNT, tDelay, LEV, STOP_LOSS_PERCENTAGE, TSL_OFFSET, volThresh, dir, mode, loadPrevDF,
                 hist_Vals, USE_INIT_AMOUNT=False):
        self.LEVERAGE = LEV
        self.INIT_AMOUNT = INIT_AMOUNT  # 10
        self.STOP_LOSS_PERCENTAGE = STOP_LOSS_PERCENTAGE
        self.savedEntryActions = []
        self.obsAction = 0
        self.hist_Vals = hist_Vals
        self.enableDoneRoutine = False

        self.runningPnL = 0
        self.percentBias = 0.3
        self.INITIAL_PRICE = 0
        self.nbBuy = 0
        self.nbHold = 0
        self.nbSell = 0

        # self.NORMOBJ = NORMOBJ
        # self.volThresh = volThresh
        self.nbTradesTrainingThresh = 100

        # lag doesn't matter
        self.BMX = BitmexTest(volThresh, dir, 100, mode, tDelay, loadPrevDF, TSL_OFFSET, hist_Vals)
        # CREATE PERPETUAL THREAD
        # self.DCL = DataCollectorLive(self.volThresh)

        return

    def reset(self):
        self.nbtradesMade = 0
        self.obsAction = 0
        self.runningPnL = 0
        self.INITIAL_PRICE = 0
        self.nbBuy = 0
        self.nbHold = 0
        self.nbSell = 0
        self.BMX.reset()
        observation = self.get_obs()

        return observation

    def setTSL(self,TSL):
        print('TSL: ' + str(TSL))
        self.BMX.setTSL(TSL)
    def setnbToIgnore(self,nbToIgnore):
        print('nbToIgnore: ' + str(nbToIgnore))
        self.BMX.setnbToIgnore(nbToIgnore)

    def step(self, action):
        self.obsAction = action
        if action == 0:
            action = 'BUY'
        if action == 1:
            action = 'HOLD'
        if action == 2:
            action = 'SELL'

        done = False
        reward = 0.0
        info = {}
        if action == 'BUY':
            # print('BUY')
            self.nbtradesMade = self.nbtradesMade + 1
            self.nbBuy = self.nbBuy + 1
            '@DELAY WHILE IN TRADE'
            # bestBid,bestAsk = self.BMX.getLatestBidAsk()
            # [entryIndex, exitIndex, entryPrice, exitPrice]
            reward, actions = self.BMX.goLong()
            self.savedEntryActions.append(actions)
            # reward = 0

        if action == 'HOLD':
            self.nbHold = self.nbHold + 1

        if action == 'SELL':
            # print('SELL')
            self.nbtradesMade = self.nbtradesMade + 1
            self.nbSell = self.nbSell + 1
            '@DELAY WHILE IN TRADE'
            # bestBid, bestAsk = self.BMX.getLatestBidAsk()
            reward, actions = self.BMX.goShort()
            self.savedEntryActions.append(actions)

            # reward = 0

        instant_pnl = reward
        self.runningPnL = self.runningPnL + instant_pnl

        if self.nbtradesMade >= self.nbTradesTrainingThresh:
            done = self.enableDoneRoutine
            info['status'] = '100 trades made.'
        else:
            done = False
            info['status'] = str(self.nbtradesMade) + ' trades made.'
        observation = self.get_obs()
        if observation == []:
            done = True
        # exit()
        return observation, reward, done, info

    def get_obs(self):

        DATA = self.BMX.getLatest1SecData()
        if DATA == []:
            return []
        x, y = DATA.shape
        DATA = DATA.reshape((1, (x * y)))
        #print('new DATA: ' + str(DATA.shape))
        #obs1 = np.asarray(DATA, dtype=float)  # latest Data
        #obs2 = np.asarray([self.obsAction], dtype=float)
        #obs = np.concatenate([obs1, obs2])
        # print ('obs ' + str(obs))
        return DATA

    def prntPerformance(self):

        ret = 'PnL: ' + str(self.runningPnL) + '\t'

        rewards = 'nbBuy : ' + str(self.nbBuy) + '\t' + \
                  'nbHold: ' + str(self.nbHold) + '\t' + \
                  'nbSell: ' + str(self.nbSell)
        #print (ret)
        print (rewards)
        epPercentagePriceChg = self.runningPnL
        log = [epPercentagePriceChg, self.nbBuy, self.nbHold, self.nbSell]
        return log

    def getEntryActionsLog(self):

        # [entryIndex, exitIndex, entryPrice, exitPrice]
        return self.savedEntryActions


class Micro_Arch_Live():

    def __init__(self, INIT_AMOUNT,PERCENT_TO_TRADE, LEV, STOP_LOSS_PERCENTAGE, TSL_OFFSET,volThresh,hist_lookback,mode, tDelay,loadPrevDF,USE_INIT_AMOUNT=False):
        self.LEVERAGE = LEV
        self.INIT_AMOUNT = INIT_AMOUNT  # 10
        self.STOP_LOSS_PERCENTAGE = STOP_LOSS_PERCENTAGE
        self.trailingSLOffset = TSL_OFFSET
        self.obsAction = 0
        self.enableDoneRoutine = False
        self.hist_lookback = hist_lookback

        self.runningPnL = 0

        self.INITIAL_PRICE = 0
        self.nbBuy = 0
        self.nbHold = 0
        self.nbSell = 0

        #self.NORMOBJ = NORMOBJ
        self.volThresh = volThresh
        self.nbTradesTrainingThresh = 100


        #CREATE PERPETUAL THREAD
        self.OML = OrderManagerLive(self.volThresh)
        if USE_INIT_AMOUNT == False:
            self.OML.setPercentageToTrade(PERCENT_TO_TRADE)
        else:
            print('not implemented')
            exit(0)

        self.OML.setTrailingOffsetValue(self.trailingSLOffset)
        #self.OML.setAvailMargin()
        self.OML.setLeverageLevel(self.LEVERAGE)
        #DATA = ['spread', 'chgMidP', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5', 'bidS_1','askS_1']
        self.DCL = DataCollectorLive(self.volThresh,self.hist_lookback,mode, tDelay,loadPrevDF)

        OML_thread = lambda: self.DCL.collectDatav3()
        thread = threading.Thread(target=OML_thread)
        thread.daemon = True
        thread.start()
        return

    def reset(self):
        self.nbtradesMade = 0
        self.obsAction = 0
        self.runningPnL = 0
        self.INITIAL_PRICE = 0
        self.nbBuy = 0
        self.nbHold = 0
        self.nbSell = 0

        observation = self.get_obs()

        return observation

    def step(self, action):
        #print('STEP: ' + str(action))
        self.obsAction = action
        if action == 0:
            action = 'BUY'
        if action == 1:
            action = 'HOLD'
        if action == 2:
            action = 'SELL'

        done = False
        reward = 0.0
        info = {}
        if action == 'BUY':
            print('BUY')
            self.nbtradesMade = self.nbtradesMade + 1
            self.nbBuy = self.nbBuy + 1
            '@DELAY WHILE IN TRADE'
            bestBid,bestAsk = self.DCL.getLatestBidAsk()
            reward = self.OML.goLong(ASK_PRICE=bestAsk)
            #reward = 0

        if action == 'HOLD':
            print('HOLD')
            self.nbHold = self.nbHold + 1

        if action == 'SELL':
            print('SELL')
            self.nbtradesMade = self.nbtradesMade + 1
            self.nbSell = self.nbSell + 1
            '@DELAY WHILE IN TRADE'
            bestBid, bestAsk = self.DCL.getLatestBidAsk()
            reward = self.OML.goShort(BID_PRICE=bestBid)
            #reward = 0

        instant_pnl = reward
        self.runningPnL = self.runningPnL + instant_pnl

        if self.nbtradesMade >=self.nbTradesTrainingThresh:
            done = self.enableDoneRoutine
            info['status'] = '100 trades made.'
        else:
            done = False
            info['status'] = str(self.nbtradesMade) + ' trades made.'
        observation = self.get_obs()
        # exit()
        return observation, reward, done, info

    def get_obs(self):
        #print('get_obs')
        DATA = self.DCL.getLatest1SecData()
        #print('DATA: ' + str(DATA))
        init = True
        while not isinstance(DATA, pd.DataFrame):
            DATA = self.DCL.getLatest1SecData()
            #print('DATA_ACTUAL_PRICE: ' + str(DATA_ACTUAL_PRICE))
            if init == True:
                print('waiting for new orderbook data')
                #print('DATA: ' + str(DATA))
                init = False
            sleep(0.001)
        DATA = DATA.as_matrix()
        x, y = DATA.shape
        DATA = DATA.reshape((1, (x * y)))

        #print('obs.shape: ' + str(DATA.shape))
        #x, y = DATA.shape
        #DATA = DATA.reshape((1, (x * y)))
        #print('new DATA: ' + str(DATA[-1]))
        #obs1 = np.asarray(DATA, dtype=float)  # latest Data
        #obs2 = np.asarray([self.obsAction], dtype=float)
        #obs = np.concatenate([obs1, obs2])
        #print ('obs1 ' + str(obs1))
        return DATA
    def prntPerformance(self):

        ret = 'PnL: ' + str(self.runningPnL) + '\t'

        rewards = 'nbBuy : ' + str(self.nbBuy) + '\t' + \
                  'nbHold: ' + str(self.nbHold) + '\t' + \
                  'nbSell: ' + str(self.nbSell)
        print (ret)
        print (rewards)
        return ret + '\n' + rewards
