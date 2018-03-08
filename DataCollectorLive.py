import ast
import csv
import pprint
import threading
from threading import Thread
import copy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from COMMON.DataNormalisation import DataNormalisation as NORM
from COMMON.SuperSmoother import TechnicalIndicators as TI
import requests as r
from datetime import datetime, timedelta
import time
import json
import logging
import os
from COMMON.API_BitMex_WebSocket import BitMEXWebsocket
from A3C_DataCollector import Bitmex

class DataCollectorLive:


    def __init__(self,volThresh,hist_lookback,mode, tDelay,loadPrevDF): #,NORMOBJ

        self.NORMOBJ = NORM(MODE='STD')
        self.dir = '/home/james/TRADING_PROJECT/DataCollector_BITMEX/BITMEX_DATA_LIVE'
        self.VolThresh = volThresh
        self.tDelay = 1.0
        LIMIT_PRICE_LVL = 1
        self.DATA = False
        self.hist_lookback = hist_lookback
        self.bid = 0
        self.ask = 0
        logger = self.setup_logger()

        BMX = Bitmex()
        lag = 200
        dir = '/home/james/TRADING_PROJECT/DataCollector_BITMEX/BITMEX_DATA'
        #DATA, DATA_ACTUAL_PRICE = BMX.loadBitmexMicroStructureDataV5(dir, 1, lag, 'training', True)
        DATA, DATA_ACTUAL_PRICE = BMX.collectDatav3(dir, lag, mode, tDelay, loadPrevDF)
        isData, NORM_DATA_1 = self.normalise(DATA)

        if isData == False:
            print('error, can not set normalisation min max from given data. input Data not found')
            exit()
        print('NORM_DATA_1.head(): ' + str(NORM_DATA_1.head()))

        self.ws = BitMEXWebsocket(endpoint="wss://www.bitmex.com/realtime", symbol="XBTUSD", api_key=None,api_secret=None)
        #self.collectDatav2()

    def collectDatav2(self):

        '''
        import pandas as pd
        df = pd.DataFrame(columns=['col1', 'col2'])
        df = df.append(pd.Series(['a', 'b'], index=['col1','col2']), ignore_index=True)
        df = df.append(pd.Series(['d', 'e'], index=['col1','col2']), ignore_index=True)
        df
          col1 col2
        0    a    b
        1    d    e
        '''
        ##DATA = ['spread', 'chgMidP', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5', 'bidS_1','askS_1']

        cols = ['time', 'chgBidP', 'chgAskP', 'chgMidP', 'bidP', 'askP', 'midP',
                'spread', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5',
                'bidS_1','askS_1']
        df = pd.DataFrame(columns=cols)
        prevBidPrice = 0
        prevAskPrice = 0
        prevMidPrice = 0
        self.latestOrderBookData = copy.deepcopy(self.ws.market_depth())
        curr_ts = datetime.strptime(self.ws.market_depth()[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
        ts = datetime.strptime(self.ws.market_depth()[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
        while (self.ws.ws.sock.connected):
            try:
                newOrderBookData = copy.deepcopy(self.ws.market_depth())
                if newOrderBookData != self.latestOrderBookData:
                    self.latestOrderBookData = newOrderBookData
                    bidP = float(newOrderBookData[0]['bids'][0][0])
                    askP = float(newOrderBookData[0]['asks'][0][0])
                    bidS_1 = float(newOrderBookData[0]['bids'][0][1])
                    askS_1 = float(newOrderBookData[0]['asks'][0][1])
                    bidS_2 = float(newOrderBookData[0]['bids'][1][1])
                    askS_2 = float(newOrderBookData[0]['asks'][1][1])
                    bidS_3 = float(newOrderBookData[0]['bids'][2][1])
                    askS_3 = float(newOrderBookData[0]['asks'][2][1])
                    bidS_4 = float(newOrderBookData[0]['bids'][3][1])
                    askS_4 = float(newOrderBookData[0]['asks'][3][1])
                    bidS_5 = float(newOrderBookData[0]['bids'][4][1])
                    askS_5 = float(newOrderBookData[0]['asks'][4][1])
                    ts = datetime.strptime(newOrderBookData[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")

                    # CALC OBI
                    OBI_LVL_1 = (bidS_1 - askS_1) / (bidS_1 + askS_1)
                    OBI_LVL_2 = (bidS_2 - askS_2) / (bidS_2 + askS_2)
                    OBI_LVL_3 = (bidS_3 - askS_3) / (bidS_3 + askS_3)
                    OBI_LVL_4 = (bidS_4 - askS_4) / (bidS_4 + askS_4)
                    OBI_LVL_5 = (bidS_5 - askS_5) / (bidS_5 + askS_5)

                    # BEST BID<ASK PRICE
                    bestMidPrice = (bidP + askP) / 2.0
                    spread = askP- bidP
                    # CALC CHGBID/ASKPRICE
                    chgBidPrice = bidP - prevBidPrice
                    chgAskPrice = askP - prevAskPrice
                    chgMidPrice = bestMidPrice - prevMidPrice
                    if prevBidPrice == 0:
                        chgBidPrice = 0
                        chgAskPrice = 0
                        chgMidPrice = 0
                    prevBidPrice = bidP
                    prevAskPrice = askP
                    prevMidPrice = bestMidPrice

                    newRow = [ts, chgBidPrice, chgAskPrice, chgMidPrice, bidP, askP, bestMidPrice,spread, OBI_LVL_1, OBI_LVL_2, OBI_LVL_3, OBI_LVL_4, OBI_LVL_5,bidS_1,askS_1]
                    df = df.append(pd.Series(newRow, index=cols), ignore_index=True)

                time.sleep(0.0001)
            except Exception as e:
                print('Error Websocket appears disconnected.' + str(e))
                exit(-1)
            #ts = datetime.strptime(self.ws.market_depth()[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
            print ('curr_ts: ' + str(curr_ts))
            print ('curr_ts + timedelta(0,self.tDelay): ' + str(curr_ts + timedelta(0,self.tDelay)))
            print ('ts: ' + str(ts))
            print ('len(df): ' + str(len(df)))

            if (curr_ts + timedelta(0,self.tDelay)) < ts :
                df = df.set_index('time')
                # DATA = ['spread', 'chgMidP', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5', 'bidS_1','askS_1']
                print ('len(df): ' + str(len(df)))

                df = df[['spread', 'chgMidP','OBI_LVL_1','OBI_LVL_2','OBI_LVL_3','OBI_LVL_4','OBI_LVL_5','bidS_1','askS_1']]
                print ('len(df): ' + str(len(df)))
                print ('df.head(20): ' + str(df.head(20)))

                DATA = df.resample(str(1) + 'S',how={'spread': 'max', 'chgMidP': 'sum',
                                                     'OBI_LVL_1': 'mean','OBI_LVL_2': 'mean',
                                                     'OBI_LVL_3': 'mean','OBI_LVL_4': 'mean',
                                                     'OBI_LVL_5': 'mean','bidS_1': 'sum','askS_1': 'sum',}).dropna(inplace=True)
                print ('DATA: ' + str(DATA))

                #reset DF
                cols = ['time', 'chgBidP', 'chgAskP', 'chgMidP', 'bidP', 'askP', 'midP',
                        'spread', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5','bidS_1','askS_1']
                df = pd.DataFrame(columns=cols)
                print(str(type(DATA)))
                isData, df_Out = self.normalise(DATA)
                if isData ==True:
                    #DATA_ACTUAL_PRICE, DATA = self.recieve1SecData(self.dir, sec=1)
                    self.DATA = df_Out
                    self.bid = float(self.ws.market_depth()[0]['bids'][0][0])
                    self.ask = float(self.ws.market_depth()[0]['asks'][0][0])

                    print('ts       : ' + str(ts))
                    print('self.DATA: ' + str(self.DATA.head()))
                    print('self.bid : ' + str(self.bid))
                    print('self.ask : ' + str(self.ask))
                else:
                    print('data did not passed vol thresh')
                    self.DATA = []
                    self.bid = float(self.ws.market_depth()[0]['bids'][0][0])
                    self.ask = float(self.ws.market_depth()[0]['asks'][0][0])
                    #data did not passed vol thresh
                curr_ts = datetime.strptime(self.ws.market_depth()[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")

            else:
                pass

    def collectDatav3(self):


        cols = ['spread', 'chgMidP','OBI_LVL_1','OBI_LVL_2','OBI_LVL_3','OBI_LVL_4','OBI_LVL_5','bidS_1','askS_1']
        #df = pd.DataFrame(columns=cols)
        #ts, chgBidPrice, chgAskPrice, chgMidPrice, bidP, askP, bestMidPrice,spread, OBI_LVL_1, OBI_LVL_2, OBI_LVL_3, OBI_LVL_4, OBI_LVL_5,bidS_1,askS_1
        #npDataTotal = np.zeros((1,14))

        spread_lst = []
        chgMidPrice_lst = []
        OBI_LVL_1_lst = []
        OBI_LVL_2_lst = []
        OBI_LVL_3_lst = []
        OBI_LVL_4_lst = []
        OBI_LVL_5_lst = []
        bidS_1_lst = []
        askS_1_lst = []
        data_lst = []
        prevBidPrice = 0
        prevAskPrice = 0
        prevMidPrice = 0
        self.latestOrderBookData = copy.deepcopy(self.ws.market_depth())
        curr_ts = datetime.strptime(self.ws.market_depth()[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
        ts = datetime.strptime(self.ws.market_depth()[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
        while (self.ws.ws.sock.connected):
            try:
                newOrderBookData = copy.deepcopy(self.ws.market_depth())
                if newOrderBookData != self.latestOrderBookData:
                    self.latestOrderBookData = newOrderBookData
                    bidP = float(newOrderBookData[0]['bids'][0][0])
                    askP = float(newOrderBookData[0]['asks'][0][0])
                    bidS_1 = float(newOrderBookData[0]['bids'][0][1])
                    askS_1 = float(newOrderBookData[0]['asks'][0][1])
                    bidS_2 = float(newOrderBookData[0]['bids'][1][1])
                    askS_2 = float(newOrderBookData[0]['asks'][1][1])
                    bidS_3 = float(newOrderBookData[0]['bids'][2][1])
                    askS_3 = float(newOrderBookData[0]['asks'][2][1])
                    bidS_4 = float(newOrderBookData[0]['bids'][3][1])
                    askS_4 = float(newOrderBookData[0]['asks'][3][1])
                    bidS_5 = float(newOrderBookData[0]['bids'][4][1])
                    askS_5 = float(newOrderBookData[0]['asks'][4][1])
                    ts = datetime.strptime(newOrderBookData[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")

                    # CALC OBI
                    OBI_LVL_1 = (bidS_1 - askS_1) / (bidS_1 + askS_1)
                    OBI_LVL_2 = (bidS_2 - askS_2) / (bidS_2 + askS_2)
                    OBI_LVL_3 = (bidS_3 - askS_3) / (bidS_3 + askS_3)
                    OBI_LVL_4 = (bidS_4 - askS_4) / (bidS_4 + askS_4)
                    OBI_LVL_5 = (bidS_5 - askS_5) / (bidS_5 + askS_5)

                    # BEST BID<ASK PRICE
                    bestMidPrice = (bidP + askP) / 2.0
                    spread = askP- bidP
                    # CALC CHGBID/ASKPRICE
                    chgBidPrice = bidP - prevBidPrice
                    chgAskPrice = askP - prevAskPrice
                    chgMidPrice = bestMidPrice - prevMidPrice
                    if prevBidPrice == 0:
                        chgBidPrice = 0
                        chgAskPrice = 0
                        chgMidPrice = 0
                    prevBidPrice = bidP
                    prevAskPrice = askP
                    prevMidPrice = bestMidPrice

                    #newRow = [spread,chgMidPrice, OBI_LVL_1, OBI_LVL_2, OBI_LVL_3, OBI_LVL_4, OBI_LVL_5,bidS_1,askS_1]
                    '''how = {'spread': 'max', 'chgMidP': 'sum',
                           'OBI_LVL_1': 'mean', 'OBI_LVL_2': 'mean',
                           'OBI_LVL_3': 'mean', 'OBI_LVL_4': 'mean',
                           'OBI_LVL_5': 'mean', 'bidS_1': 'sum', 'askS_1': 'sum', }
                    '''
                    spread_lst.append(spread)
                    chgMidPrice_lst.append(chgMidPrice)
                    OBI_LVL_1_lst.append(OBI_LVL_1)
                    OBI_LVL_2_lst.append(OBI_LVL_2)
                    OBI_LVL_3_lst.append(OBI_LVL_3)
                    OBI_LVL_4_lst.append(OBI_LVL_4)
                    OBI_LVL_5_lst.append(OBI_LVL_5)
                    bidS_1_lst.append(bidS_1)
                    askS_1_lst.append(askS_1)


                time.sleep(0.0001)
            except Exception as e:
                print('Error Websocket appears disconnected.' + str(e))
                exit(-1)

            if (curr_ts + timedelta(0,self.tDelay)) < ts :
                '''
                how = {'spread': 'max', 'chgMidP': 'sum',
                       'OBI_LVL_1': 'mean', 'OBI_LVL_2': 'mean',
                       'OBI_LVL_3': 'mean', 'OBI_LVL_4': 'mean',
                       'OBI_LVL_5': 'mean', 'bidS_1': 'sum', 'askS_1': 'sum', }
                '''
                '''
                print('spread_lst: ' + str(spread_lst))
                print('chgMidPrice_lst: ' + str(chgMidPrice_lst))
                print('OBI_LVL_1_lst: ' + str(OBI_LVL_1_lst))
                print('OBI_LVL_2_lst: ' + str(OBI_LVL_2_lst))
                print('OBI_LVL_3_lst: ' + str(OBI_LVL_3_lst))
                print('OBI_LVL_4_lst: ' + str(OBI_LVL_4_lst))
                print('OBI_LVL_5_lst: ' + str(OBI_LVL_5_lst))
                print('bidS_1_lst: ' + str(bidS_1_lst))
                print('askS_1_lst: ' + str(askS_1_lst))
'''
                spread_New = np.asarray(spread_lst).mean()
                chgMidPrice_New = np.asarray(chgMidPrice_lst).cumsum()[-1]
                OBI_LVL_1_New = np.asarray(OBI_LVL_1_lst).mean()
                OBI_LVL_2_New = np.asarray(OBI_LVL_2_lst).mean()
                OBI_LVL_3_New = np.asarray(OBI_LVL_3_lst).mean()
                OBI_LVL_4_New = np.asarray(OBI_LVL_4_lst).mean()
                OBI_LVL_5_New = np.asarray(OBI_LVL_5_lst).mean()
                bidS_1_New = np.asarray(bidS_1_lst).mean()
                askS_1_New = np.asarray(askS_1_lst).mean()
                '''
                print('spread_New: ' + str(spread_New))
                print('chgMidPrice_New: ' + str(chgMidPrice_New))
                print('OBI_LVL_1_New: ' + str(OBI_LVL_1_New))
                print('OBI_LVL_2_New: ' + str(OBI_LVL_2_New))
                print('OBI_LVL_3_New: ' + str(OBI_LVL_3_New))
                print('OBI_LVL_4_New: ' + str(OBI_LVL_4_New))
                print('OBI_LVL_5_New: ' + str(OBI_LVL_5_New))
                print('bidS_1_New: ' + str(bidS_1_New))
                print('askS_1_New: ' + str(askS_1_New))
                '''
                #['spread', 'chgMidP','OBI_LVL_1','OBI_LVL_2','OBI_LVL_3','OBI_LVL_4','OBI_LVL_5','bidS_1','askS_1']
                #npData = np.zeros((9,9),dtype=float)
                data = [spread_New,
                        chgMidPrice_New,
                        OBI_LVL_1_New,
                        OBI_LVL_2_New,
                        OBI_LVL_3_New,
                        OBI_LVL_4_New,
                        OBI_LVL_5_New,
                        bidS_1_New,
                        askS_1_New]

                data_lst.append(data)
                # df = df.append(data,ignore_index=True)
                # df_actual = df_actual.append(data_actual,ignore_index=True)
                # df.dropna(inplace=True)

                spread_lst = []
                chgMidPrice_lst = []
                OBI_LVL_1_lst = []
                OBI_LVL_2_lst = []
                OBI_LVL_3_lst = []
                OBI_LVL_4_lst = []
                OBI_LVL_5_lst = []
                bidS_1_lst = []
                askS_1_lst = []
                #print ('len(data_lst) before: ' + str(len(data_lst)))
                if len(data_lst) > self.hist_lookback:
                    data_lst = data_lst[-self.hist_lookback:]
                    #print ('self.hist_lookback: ' + str(self.hist_lookback))

                    #print ('len(data_lst) after: ' +str(len(data_lst)))
                    npData = np.asarray(data_lst)
                    #print ('npData: ' + str(npData.shape))
                    df = pd.DataFrame(data=npData, columns=cols)

                    #print('df: ' + str(df))
                    '''
                    data = [[np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN]]
                            df = pd.DataFrame(data=data, columns=cols)
                            df.set_value(0,'spread',spread_New)
                            df.set_value(0, 'chgMidP', chgMidPrice_New)
                            df.set_value(0, 'OBI_LVL_1', OBI_LVL_1_New)
                            df.set_value(0, 'OBI_LVL_2', OBI_LVL_2_New)
                            df.set_value(0, 'OBI_LVL_3', OBI_LVL_3_New)
                            df.set_value(0, 'OBI_LVL_4', OBI_LVL_4_New)
                            df.set_value(0, 'OBI_LVL_5', OBI_LVL_5_New)
                            df.set_value(0, 'bidS_1', bidS_1_New)
                            df.set_value(0, 'askS_1', askS_1_New)
                    
                            df.dropna(inplace=True)
                            '''
                    #print('NEW DF: ' + str(df))
                    isData, df_Out = self.normalise(df)
                    if isData ==True:
                        print('Data passed vol thresh')

                        self.DATA = df_Out
                        self.bid = float(self.ws.market_depth()[0]['bids'][0][0])
                        self.ask = float(self.ws.market_depth()[0]['asks'][0][0])

                        #print('ts       : ' + str(ts))
                        #print('self.DATA: ' + str(self.DATA.head(1)))
                        #print('self.bid : ' + str(self.bid))
                        #print('self.ask : ' + str(self.ask))
                    else:
                        print('data did not passed vol thresh')
                        self.DATA = False
                        self.bid = float(self.ws.market_depth()[0]['bids'][0][0])
                        self.ask = float(self.ws.market_depth()[0]['asks'][0][0])
                        #data did not passed vol thresh
                curr_ts = datetime.strptime(self.ws.market_depth()[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")

            else:
                pass

    def setup_logger(self):
        # Prints logger info to terminal
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Change this to DEBUG if you want a lot more info
        ch = logging.StreamHandler()
        # create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        # add formatter to ch
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def normalise(self,df):

        newNpData = []
        newNpDataIndexes = []
        npData = df.as_matrix()
        indexVals = df.index.values
        for x in range(0, len(npData)):
            if npData[x][7] > self.VolThresh or npData[x][8] > self.VolThresh:
                newNpData.append(npData[x])
                newNpDataIndexes.append(indexVals[x])
        # npData = np.zeros((len(newNpData),7),dtype=float)
        if newNpData != []:
            npData = np.asarray(newNpData, dtype=float)
            df_Out = pd.DataFrame(data=npData, columns=df.columns, index=newNpDataIndexes, dtype=np.float32)
            #NORM
            df_Out = self.NORMOBJ.IN(df_Out, isLog=False)
            return True,df_Out
        else:
            return False,''

    def getLatestBidAsk(self):

        return self.bid,self.ask

    def getLatest1SecData(self):
        newData = self.DATA
        self.DATA = False
        return newData

class DataCollectorLiveTest:


    def __init__(self,volThresh): #,NORMOBJ

        self.NORMOBJ = NORM(MODE='STD')
        #self.dir = '/home/james/TRADING_PROJECT/DataCollector_BITMEX/BITMEX_DATA_LIVE'
        self.VolThresh = volThresh
        self.tDelay = 1.0
        LIMIT_PRICE_LVL = 1
        self.DATA = False
        self.bid = 0
        self.ask = 0
        logger = self.setup_logger()

        BMX = Bitmex()
        lag = 200
        dir = '/home/james/TRADING_PROJECT/DataCollector_BITMEX/BITMEX_DATA'
        DATA, DATA_ACTUAL_PRICE = BMX.collectDatav3(dir, 1, lag, 'training', False)
        isData, NORM_DATA_1 = self.normalise(DATA)

        if isData == False:
            print('error, can not set normalisation min max from given data. input Data not found')
            exit()
        print('NORM_DATA_1.head(): ' + str(NORM_DATA_1.head()))

        self.ws = BitMEXWebsocket(endpoint="wss://www.bitmex.com/realtime", symbol="XBTUSD", api_key=None,api_secret=None)
        #self.collectDatav2()

    def collectDatav2(self):

        '''
        import pandas as pd
        df = pd.DataFrame(columns=['col1', 'col2'])
        df = df.append(pd.Series(['a', 'b'], index=['col1','col2']), ignore_index=True)
        df = df.append(pd.Series(['d', 'e'], index=['col1','col2']), ignore_index=True)
        df
          col1 col2
        0    a    b
        1    d    e
        '''
        ##DATA = ['spread', 'chgMidP', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5', 'bidS_1','askS_1']

        cols = ['time', 'chgBidP', 'chgAskP', 'chgMidP', 'bidP', 'askP', 'midP',
                'spread', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5',
                'bidS_1','askS_1']
        df = pd.DataFrame(columns=cols)
        prevBidPrice = 0
        prevAskPrice = 0
        prevMidPrice = 0
        self.latestOrderBookData = copy.deepcopy(self.ws.market_depth())
        curr_ts = datetime.strptime(self.ws.market_depth()[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
        ts = datetime.strptime(self.ws.market_depth()[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
        while (self.ws.ws.sock.connected):
            try:
                newOrderBookData = copy.deepcopy(self.ws.market_depth())
                if newOrderBookData != self.latestOrderBookData:
                    self.latestOrderBookData = newOrderBookData
                    bidP = float(newOrderBookData[0]['bids'][0][0])
                    askP = float(newOrderBookData[0]['asks'][0][0])
                    bidS_1 = float(newOrderBookData[0]['bids'][0][1])
                    askS_1 = float(newOrderBookData[0]['asks'][0][1])
                    bidS_2 = float(newOrderBookData[0]['bids'][1][1])
                    askS_2 = float(newOrderBookData[0]['asks'][1][1])
                    bidS_3 = float(newOrderBookData[0]['bids'][2][1])
                    askS_3 = float(newOrderBookData[0]['asks'][2][1])
                    bidS_4 = float(newOrderBookData[0]['bids'][3][1])
                    askS_4 = float(newOrderBookData[0]['asks'][3][1])
                    bidS_5 = float(newOrderBookData[0]['bids'][4][1])
                    askS_5 = float(newOrderBookData[0]['asks'][4][1])
                    ts = datetime.strptime(newOrderBookData[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")

                    # CALC OBI
                    OBI_LVL_1 = (bidS_1 - askS_1) / (bidS_1 + askS_1)
                    OBI_LVL_2 = (bidS_2 - askS_2) / (bidS_2 + askS_2)
                    OBI_LVL_3 = (bidS_3 - askS_3) / (bidS_3 + askS_3)
                    OBI_LVL_4 = (bidS_4 - askS_4) / (bidS_4 + askS_4)
                    OBI_LVL_5 = (bidS_5 - askS_5) / (bidS_5 + askS_5)

                    # BEST BID<ASK PRICE
                    bestMidPrice = (bidP + askP) / 2.0
                    spread = askP- bidP
                    # CALC CHGBID/ASKPRICE
                    chgBidPrice = bidP - prevBidPrice
                    chgAskPrice = askP - prevAskPrice
                    chgMidPrice = bestMidPrice - prevMidPrice
                    if prevBidPrice == 0:
                        chgBidPrice = 0
                        chgAskPrice = 0
                        chgMidPrice = 0
                    prevBidPrice = bidP
                    prevAskPrice = askP
                    prevMidPrice = bestMidPrice

                    newRow = [ts, chgBidPrice, chgAskPrice, chgMidPrice, bidP, askP, bestMidPrice,spread, OBI_LVL_1, OBI_LVL_2, OBI_LVL_3, OBI_LVL_4, OBI_LVL_5,bidS_1,askS_1]
                    df = df.append(pd.Series(newRow, index=cols), ignore_index=True)

                time.sleep(0.0001)
            except Exception as e:
                print('Error Websocket appears disconnected.' + str(e))
                exit(-1)
            #ts = datetime.strptime(self.ws.market_depth()[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
            print ('curr_ts: ' + str(curr_ts))
            print ('curr_ts + timedelta(0,self.tDelay): ' + str(curr_ts + timedelta(0,self.tDelay)))
            print ('ts: ' + str(ts))
            print ('len(df): ' + str(len(df)))

            if (curr_ts + timedelta(0,self.tDelay)) < ts :
                df = df.set_index('time')
                # DATA = ['spread', 'chgMidP', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5', 'bidS_1','askS_1']
                print ('len(df): ' + str(len(df)))

                df = df[['spread', 'chgMidP','OBI_LVL_1','OBI_LVL_2','OBI_LVL_3','OBI_LVL_4','OBI_LVL_5','bidS_1','askS_1']]
                print ('len(df): ' + str(len(df)))
                print ('df.head(20): ' + str(df.head(20)))

                DATA = df.resample(str(1) + 'S',how={'spread': 'max', 'chgMidP': 'sum',
                                                     'OBI_LVL_1': 'mean','OBI_LVL_2': 'mean',
                                                     'OBI_LVL_3': 'mean','OBI_LVL_4': 'mean',
                                                     'OBI_LVL_5': 'mean','bidS_1': 'sum','askS_1': 'sum',}).dropna(inplace=True)
                print ('DATA: ' + str(DATA))

                #reset DF
                cols = ['time', 'chgBidP', 'chgAskP', 'chgMidP', 'bidP', 'askP', 'midP',
                        'spread', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5','bidS_1','askS_1']
                df = pd.DataFrame(columns=cols)
                print(str(type(DATA)))
                isData, df_Out = self.normalise(DATA)
                if isData ==True:
                    #DATA_ACTUAL_PRICE, DATA = self.recieve1SecData(self.dir, sec=1)
                    self.DATA = df_Out
                    self.bid = float(self.ws.market_depth()[0]['bids'][0][0])
                    self.ask = float(self.ws.market_depth()[0]['asks'][0][0])

                    print('ts       : ' + str(ts))
                    print('self.DATA: ' + str(self.DATA.head()))
                    print('self.bid : ' + str(self.bid))
                    print('self.ask : ' + str(self.ask))
                else:
                    print('data did not passed vol thresh')
                    self.DATA = []
                    self.bid = float(self.ws.market_depth()[0]['bids'][0][0])
                    self.ask = float(self.ws.market_depth()[0]['asks'][0][0])
                    #data did not passed vol thresh
                curr_ts = datetime.strptime(self.ws.market_depth()[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")

            else:
                pass

    def collectDatav3(self):


        cols = ['spread', 'chgMidP','OBI_LVL_1','OBI_LVL_2','OBI_LVL_3','OBI_LVL_4','OBI_LVL_5','bidS_1','askS_1']
        #df = pd.DataFrame(columns=cols)
        #ts, chgBidPrice, chgAskPrice, chgMidPrice, bidP, askP, bestMidPrice,spread, OBI_LVL_1, OBI_LVL_2, OBI_LVL_3, OBI_LVL_4, OBI_LVL_5,bidS_1,askS_1
        #npDataTotal = np.zeros((1,14))

        spread_lst = []
        chgMidPrice_lst = []
        OBI_LVL_1_lst = []
        OBI_LVL_2_lst = []
        OBI_LVL_3_lst = []
        OBI_LVL_4_lst = []
        OBI_LVL_5_lst = []
        bidS_1_lst = []
        askS_1_lst = []

        prevBidPrice = 0
        prevAskPrice = 0
        prevMidPrice = 0
        self.latestOrderBookData = copy.deepcopy(self.ws.market_depth())
        curr_ts = datetime.strptime(self.ws.market_depth()[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
        ts = datetime.strptime(self.ws.market_depth()[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
        while (self.ws.ws.sock.connected):
            try:
                newOrderBookData = copy.deepcopy(self.ws.market_depth())
                if newOrderBookData != self.latestOrderBookData:
                    self.latestOrderBookData = newOrderBookData
                    bidP = float(newOrderBookData[0]['bids'][0][0])
                    askP = float(newOrderBookData[0]['asks'][0][0])
                    bidS_1 = float(newOrderBookData[0]['bids'][0][1])
                    askS_1 = float(newOrderBookData[0]['asks'][0][1])
                    bidS_2 = float(newOrderBookData[0]['bids'][1][1])
                    askS_2 = float(newOrderBookData[0]['asks'][1][1])
                    bidS_3 = float(newOrderBookData[0]['bids'][2][1])
                    askS_3 = float(newOrderBookData[0]['asks'][2][1])
                    bidS_4 = float(newOrderBookData[0]['bids'][3][1])
                    askS_4 = float(newOrderBookData[0]['asks'][3][1])
                    bidS_5 = float(newOrderBookData[0]['bids'][4][1])
                    askS_5 = float(newOrderBookData[0]['asks'][4][1])
                    ts = datetime.strptime(newOrderBookData[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")

                    # CALC OBI
                    OBI_LVL_1 = (bidS_1 - askS_1) / (bidS_1 + askS_1)
                    OBI_LVL_2 = (bidS_2 - askS_2) / (bidS_2 + askS_2)
                    OBI_LVL_3 = (bidS_3 - askS_3) / (bidS_3 + askS_3)
                    OBI_LVL_4 = (bidS_4 - askS_4) / (bidS_4 + askS_4)
                    OBI_LVL_5 = (bidS_5 - askS_5) / (bidS_5 + askS_5)

                    # BEST BID<ASK PRICE
                    bestMidPrice = (bidP + askP) / 2.0
                    spread = askP- bidP
                    # CALC CHGBID/ASKPRICE
                    chgBidPrice = bidP - prevBidPrice
                    chgAskPrice = askP - prevAskPrice
                    chgMidPrice = bestMidPrice - prevMidPrice
                    if prevBidPrice == 0:
                        chgBidPrice = 0
                        chgAskPrice = 0
                        chgMidPrice = 0
                    prevBidPrice = bidP
                    prevAskPrice = askP
                    prevMidPrice = bestMidPrice

                    #newRow = [spread,chgMidPrice, OBI_LVL_1, OBI_LVL_2, OBI_LVL_3, OBI_LVL_4, OBI_LVL_5,bidS_1,askS_1]
                    '''how = {'spread': 'max', 'chgMidP': 'sum',
                           'OBI_LVL_1': 'mean', 'OBI_LVL_2': 'mean',
                           'OBI_LVL_3': 'mean', 'OBI_LVL_4': 'mean',
                           'OBI_LVL_5': 'mean', 'bidS_1': 'sum', 'askS_1': 'sum', }
                    '''
                    spread_lst.append(spread)
                    chgMidPrice_lst.append(chgMidPrice)
                    OBI_LVL_1_lst.append(OBI_LVL_1)
                    OBI_LVL_2_lst.append(OBI_LVL_2)
                    OBI_LVL_3_lst.append(OBI_LVL_3)
                    OBI_LVL_4_lst.append(OBI_LVL_4)
                    OBI_LVL_5_lst.append(OBI_LVL_5)
                    bidS_1_lst.append(bidS_1)
                    askS_1_lst.append(askS_1)


                time.sleep(0.0001)
            except Exception as e:
                print('Error Websocket appears disconnected.' + str(e))
                exit(-1)

            if (curr_ts + timedelta(0,self.tDelay)) < ts :
                '''
                how = {'spread': 'max', 'chgMidP': 'sum',
                       'OBI_LVL_1': 'mean', 'OBI_LVL_2': 'mean',
                       'OBI_LVL_3': 'mean', 'OBI_LVL_4': 'mean',
                       'OBI_LVL_5': 'mean', 'bidS_1': 'sum', 'askS_1': 'sum', }
                '''
                '''
                print('spread_lst: ' + str(spread_lst))
                print('chgMidPrice_lst: ' + str(chgMidPrice_lst))
                print('OBI_LVL_1_lst: ' + str(OBI_LVL_1_lst))
                print('OBI_LVL_2_lst: ' + str(OBI_LVL_2_lst))
                print('OBI_LVL_3_lst: ' + str(OBI_LVL_3_lst))
                print('OBI_LVL_4_lst: ' + str(OBI_LVL_4_lst))
                print('OBI_LVL_5_lst: ' + str(OBI_LVL_5_lst))
                print('bidS_1_lst: ' + str(bidS_1_lst))
                print('askS_1_lst: ' + str(askS_1_lst))
'''
                spread_New = np.asarray(spread_lst).mean()
                chgMidPrice_New = np.asarray(chgMidPrice_lst).cumsum()[-1]
                OBI_LVL_1_New = np.asarray(OBI_LVL_1_lst).mean()
                OBI_LVL_2_New = np.asarray(OBI_LVL_2_lst).mean()
                OBI_LVL_3_New = np.asarray(OBI_LVL_3_lst).mean()
                OBI_LVL_4_New = np.asarray(OBI_LVL_4_lst).mean()
                OBI_LVL_5_New = np.asarray(OBI_LVL_5_lst).mean()
                bidS_1_New = np.asarray(bidS_1_lst).mean()
                askS_1_New = np.asarray(askS_1_lst).mean()
                '''
                print('spread_New: ' + str(spread_New))
                print('chgMidPrice_New: ' + str(chgMidPrice_New))
                print('OBI_LVL_1_New: ' + str(OBI_LVL_1_New))
                print('OBI_LVL_2_New: ' + str(OBI_LVL_2_New))
                print('OBI_LVL_3_New: ' + str(OBI_LVL_3_New))
                print('OBI_LVL_4_New: ' + str(OBI_LVL_4_New))
                print('OBI_LVL_5_New: ' + str(OBI_LVL_5_New))
                print('bidS_1_New: ' + str(bidS_1_New))
                print('askS_1_New: ' + str(askS_1_New))
                '''
                #['spread', 'chgMidP','OBI_LVL_1','OBI_LVL_2','OBI_LVL_3','OBI_LVL_4','OBI_LVL_5','bidS_1','askS_1']
                #npData = np.zeros((9,9),dtype=float)
                data = [[np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN]]
                df = pd.DataFrame(data=data, columns=cols)
                df.set_value(0,'spread',spread_New)
                df.set_value(0, 'chgMidP', chgMidPrice_New)
                df.set_value(0, 'OBI_LVL_1', OBI_LVL_1_New)
                df.set_value(0, 'OBI_LVL_2', OBI_LVL_2_New)
                df.set_value(0, 'OBI_LVL_3', OBI_LVL_3_New)
                df.set_value(0, 'OBI_LVL_4', OBI_LVL_4_New)
                df.set_value(0, 'OBI_LVL_5', OBI_LVL_5_New)
                df.set_value(0, 'bidS_1', bidS_1_New)
                df.set_value(0, 'askS_1', askS_1_New)

                df.dropna(inplace=True)
                print('NEW DF: ' + str(df))
                isData, df_Out = self.normalise(df)
                if isData ==True:
                    print('Data passed vol thresh')

                    self.DATA = df_Out
                    self.bid = float(self.ws.market_depth()[0]['bids'][0][0])
                    self.ask = float(self.ws.market_depth()[0]['asks'][0][0])

                    print('ts       : ' + str(ts))
                    print('self.DATA: ' + str(self.DATA.head()))
                    print('self.bid : ' + str(self.bid))
                    print('self.ask : ' + str(self.ask))
                else:
                    print('data did not passed vol thresh')
                    self.DATA = False
                    self.bid = float(self.ws.market_depth()[0]['bids'][0][0])
                    self.ask = float(self.ws.market_depth()[0]['asks'][0][0])
                    #data did not passed vol thresh
                curr_ts = datetime.strptime(self.ws.market_depth()[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")

            else:
                pass

    def setup_logger(self):
        # Prints logger info to terminal
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Change this to DEBUG if you want a lot more info
        ch = logging.StreamHandler()
        # create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        # add formatter to ch
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def normalise(self,df):

        newNpData = []
        newNpDataIndexes = []
        npData = df.as_matrix()
        indexVals = df.index.values
        for x in range(0, len(npData)):
            if npData[x][7] > self.VolThresh or npData[x][8] > self.VolThresh:
                newNpData.append(npData[x])
                newNpDataIndexes.append(indexVals[x])
        # npData = np.zeros((len(newNpData),7),dtype=float)
        if newNpData != []:
            npData = np.asarray(newNpData, dtype=float)
            df_Out = pd.DataFrame(data=npData, columns=df.columns, index=newNpDataIndexes, dtype=np.float32)
            #NORM
            df_Out = self.NORMOBJ.IN(df_Out, isLog=False)
            return True,df_Out
        else:
            return False,''

    def getLatestBidAsk(self):

        return self.bid,self.ask

    def getLatest1SecData(self):

        return self.DATA