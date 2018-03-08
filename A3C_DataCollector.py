import ast
import csv
import pprint
import threading
from threading import Thread

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DataNormalisation import DataNormalisation as NORM
from SuperSmoother import TechnicalIndicators as TI
import requests as r
from datetime import datetime, timedelta
import time
import json
import os
futuredate = datetime.now() + timedelta(days=10)
class Bitmex:
    def __init__(self):
        # path = 'BTC_USD.txt'
        # self.getAllData(windowSize=20)
        #nbLatestMins = 3
        #self.getLatest1mDataBitmex(nbLatestMins, windowSize=20)
        #self.loadBitmexMicroStructureData('./bitmex_data','TRAINING')
        '''
        'DataCollector_BITMEX/BITMEX_DATA/OrderBook.txt'
        self.df_OB = []
        self.df_TR = []
        print ('os.getcwd(): ' +str(os.getcwd()))
        df = self.loadBitmexMicroStructureDataV4('/home/james/TRADING_PROJECT/DataCollector_BITMEX/BITMEX_DATA',1,'TRAINING')
        print (df.head(50))
        plt.plot(df[['RT']][0:200].as_matrix())
        plt.plot(df[['MidPrice_1']][0:200].as_matrix())

        plt.show()
        plt.hold(True)
        return
        '''

        self.TSLOffset = 5
        self.exitStrat = 0
    def splitdf(self,df):

        npData=df.as_matrix()
        nbRows=len(npData)
        nbSplitRows=0
        if (nbRows % 2) == 0:
            nbSplitRows = nbRows/2
        else:
            nbSplitRows = (nbRows-1) / 2

        df_1 = pd.DataFrame(data=npData[0:nbSplitRows], columns=df.columns, index=df.index[0:nbSplitRows], dtype=np.float32)
        df_2 = pd.DataFrame(data=npData[nbSplitRows:], columns=df.columns, index=df.index[nbSplitRows:],dtype=np.float32)

        return df_1,df_2
    def getLatest1mDataBitmex(self,resampRate, nbLatestDays, windowSize,orig_p_ss_period=6,delay=2):
        # past seven days
        SS = TI()
        # print(datetime.utcnow())
        ss = 'xx'
        utcTime = 99
        while ss != '30':
            utcTime = datetime.utcnow() - timedelta(hours=8 * 3 * nbLatestDays)
            ss = str(utcTime).split(' ')[1].split(':')[2].split('.')[0]
            print('ss: ' + str(ss))
            time.sleep(0.2)

            # utcTime = utcTime + datetime.timedelta(hours=9)
        utcTime = utcTime + timedelta(seconds=30)
        df = pd.date_range(utcTime, periods=(3 * nbLatestDays), freq='8H')
        print('utcTime: ' + str(utcTime))
        lstAll = []
        indexlst = []

        i = 0
        for t in df:

            YY = str(t).split(' ')[0].split('-')[0]
            MM = str(t).split(' ')[0].split('-')[1]
            DD = str(t).split(' ')[0].split('-')[2]
            hh = str(t).split(' ')[1].split(':')[0]
            mm = str(t).split(' ')[1].split(':')[1]
            ss = str(t).split(' ')[1].split(':')[2].split('.')[0]
            # uuu = str(t).split(' ')[1].split(':')[2].split('.')[1][0:3]
            uuu = '000'
            t = t + timedelta(hours=8)
            YY2 = str(t).split(' ')[0].split('-')[0]
            MM2 = str(t).split(' ')[0].split('-')[1]
            DD2 = str(t).split(' ')[0].split('-')[2]
            hh2 = str(t).split(' ')[1].split(':')[0]
            mm2 = str(t).split(' ')[1].split(':')[1]
            ss2 = str(t).split(' ')[1].split(':')[2].split('.')[0]
            # uuu2 = str(t).split(' ')[1].split(':')[2].split('.')[1][0:3]
            uuu2 = '000'
            startTime = YY + '-' + MM + '-' + DD + 'T' + hh + '%3A' + mm + '%3A' + ss + '.' + uuu + 'Z'
            endTime = YY2 + '-' + MM2 + '-' + DD2 + 'T' + hh2 + '%3A' + mm2 + '%3A' + ss2 + '.' + uuu2 + 'Z'
            print('startTime: ' + startTime)
            print('endTime  : ' + endTime)
            url = 'https://www.bitmex.com/api/v1/quote/bucketed?' \
                  'binSize=1m' \
                  '&partial=false' \
                  '&symbol=XBT' \
                  '&count=480' \
                  '&reverse=false' \
                  '&startTime=' + str(startTime) + \
                  '&endTime=' + str(endTime)
            print('url: ' + str(url))
            response = r.get(url)
            print('status_code: ' + str(response.status_code))
            j = response.json()
            # print(j)

            for p in j:
                lst = []
                ts = datetime.strptime(p['timestamp'][0:-1], "%Y-%m-%dT%H:%M:%S.%f")
                #print ('ts: ' +str(ts))
                indexlst.append(ts)
                lst.append(p['bidPrice'])
                lst.append(p['askPrice'])
                lst.append(p['bidSize'])
                lst.append(p['askSize'])
                lstAll.append(lst)
            print(str(i))
            i = i + 1
            time.sleep(delay)

        npORIG = np.asarray(lstAll, dtype=float)
        # print(npORIG)


        cols = ['bidPrice', 'askPrice']
        colsV = ['bidSize', 'askSize']
        npORIGV = npORIG[:, 2:4]
        npORIG = npORIG[:, 0:2]
        # print(npORIG)
        # print(npORIGV)

        dfORIGSS = pd.DataFrame(npORIG, columns=cols, dtype=float, index=indexlst)  #
        dfORIGSS.index.name = 'time'
        dfORIGSSV = pd.DataFrame(npORIGV, columns=colsV, dtype=float, index=indexlst)  #
        dfORIGSSV.index.name = 'time'

        dfORIGSS = dfORIGSS.resample(resampRate, axis=0)        # chg point tDelta
        dfORIGSSV = dfORIGSSV.resample(resampRate, axis=0)      # chg point tDelta

        NM_P = NORM(MODE='ZSCORE')          #gives rol mean rol stddev and zscore
        NM_V = NORM(MODE='ZSCORE')

        NM_P_DIFF = NORM(MODE='DIFF')  # gives second order data
        NM_V_DIFF = NORM(MODE='DIFF')

        df_p = NM_P.IN(dfORIGSS, windowSize=windowSize)
        df_v = NM_V.IN(dfORIGSSV, windowSize=windowSize)

        df_p_diff = NM_P_DIFF.IN(dfORIGSS, windowSize=windowSize)
        df_v_diff = NM_V_DIFF.IN(dfORIGSSV, windowSize=windowSize)

        #dfSSDIFF_Price.plot()
        #dfSSSTDDEV_Price.plot()
        #plt.show()
        #plt.hold()
        # dfSSSTD_Price.to_csv('/home/james/RL/bitmex_data/'+str('STD_Price_')+str(df[0])+str('.csv'))
        # dfSSDIFF_Price.to_csv('/home/james/RL/bitmex_data/'+str('DIFF_Price-')+str(df[0])+str('.csv'))
        # dfSSSTDDEV_Price.to_csv('/home/james/RL/bitmex_data/'+str('STDDEV_Price_')+str(df[0])+str('.csv'))
        # dfSSSTD_Vol.to_csv('/home/james/RL/bitmex_data/'+str('STD_Vol_')+str(df[0])+str('.csv'))
        df_pv = pd.concat([df_p,df_v,df_p_diff,df_v_diff], axis=1)

        dfORIGSS_out = SS.LP_Filter(dfORIGSS, period=orig_p_ss_period)
        return dfORIGSS_out, df_pv

    def getOrders(self):

        #To get open orders only, send {"open": true} in the filter param.
        url = 'https://testnet.bitmex.com/api/v1/order?count=100&reverse=false'

    def createNewOrder(self):
        url = 'https://testnet.bitmex.com/api/v1/order?symbol=XBTUSD&' \
              'side=Buy&' +\
              'displayQty=0&' +\
              'ordType=Market&'+ \
              'orderQty=1&' +\
              'timeInForce=GoodTillCancel'
        # Order quantity in units of the instrument (i.e. contracts).



        return

    def getNewBitmexDF_Files(self, dir, nbLatestDays, windowSize, resampRate,ss_period):
        # past 7 days. each data point reps 1 min
        ORIG_P, NORM_PV = self.getLatest1mDataBitmex(resampRate, nbLatestDays, windowSize, ss_period)
        # V ['bidSize_MEAN', 'askSize_MEAN', 'bidSize_STDDEV', 'askSize_STDDEV', 'bidSize_ZSCORE', 'askSize_ZSCORE']
        print('ORIG_P : ' +str(len(ORIG_P)))
        print('NORM_PV: ' + str(len(NORM_PV)))
        ORIG_P.to_csv(dir + '/ORIG_P_NEW.csv', index=True)
        NORM_PV.to_csv(dir + '/NORM_PV_NEW.csv', index=True)

        ORIG_P_1, ORIG_P_2 = self.splitdf(ORIG_P)  # ORIGINAL PRICES USED IN ENV
        NORM_PV_1, NORM_PV_2 = self.splitdf(NORM_PV)  # NORMALISED PRICE AND VOLS with mean stddev and zscore ALL THE OBSERVABLE INFOMATION !!!

        ORIG_P_1.to_csv(dir + '/ORIG_P_1.csv', index=True)
        ORIG_P_2.to_csv(dir + '/ORIG_P_2.csv', index=True)

        NORM_PV_1.to_csv(dir + '/NORM_PV_1.csv', index=True)
        NORM_PV_2.to_csv(dir + '/NORM_PV_2.csv', index=True)
        return [ORIG_P_1, ORIG_P_2], [NORM_PV_1, NORM_PV_2]

    def loadBitmexDF_Files(self, dir, tDelta, nbSamp):
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

        ORIG_P_1 = pd.read_csv(dir + '/ORIG_P_1.csv', index_col=['time'], parse_dates=['time'], date_parser=dateparse)
        ORIG_P_2 = pd.read_csv(dir + '/ORIG_P_2.csv', index_col=['time'], parse_dates=['time'], date_parser=dateparse)

        NORM_PV_1 = pd.read_csv(dir + '/NORM_PV_1.csv', index_col=['time'], parse_dates=['time'], date_parser=dateparse)
        NORM_PV_2 = pd.read_csv(dir + '/NORM_PV_2.csv', index_col=['time'], parse_dates=['time'], date_parser=dateparse)

        # NORM_PV_1['bidSize_ZSCORE'].plot()
        # NORM_PV_1['askSize_STDDEV'].plot()
        # plt.show()
        # exit()
        ORIG_P_OS_LST_1  = self.oversample_df_OHLCv2(ORIG_P_1, tDelta, nbSamp)  # 10 mins
        ORIG_P_OS_LST_2  = self.oversample_df_OHLCv2(ORIG_P_2, tDelta, nbSamp)  # 10 mins
        NORM_PV_OS_LST_1 = self.oversample_df_OHLCv2(NORM_PV_1, tDelta, nbSamp)  # 10 mins
        NORM_PV_OS_LST_2 = self.oversample_df_OHLCv2(NORM_PV_2, tDelta, nbSamp)  # 10 mins


        return [ORIG_P_OS_LST_1, ORIG_P_OS_LST_2], [NORM_PV_OS_LST_1, NORM_PV_OS_LST_2]
    def loadBitmexDFFilesNew(self, dir, tDelta, nbSamp):
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

        ORIG_P_1 = pd.read_csv(dir + '/ORIG_P_NEW.csv', index_col=['time'], parse_dates=['time'], date_parser=dateparse)
        NORM_PV_1 = pd.read_csv(dir + '/NORM_PV_NEW.csv', index_col=['time'], parse_dates=['time'], date_parser=dateparse)

        NORM_PV_1 = NORM_PV_1[[
            'bidPrice_MEAN',
            'askPrice_MEAN',
            #'bidSize_MEAN',
            #'askSize_MEAN',
        ]]

        #NORM_PV_1['askPrice_ZSCORE'].plot()
        #time,
        # bidPrice_MEAN,
        # askPrice_MEAN,
        # bidPrice_STDDEV,
        # askPrice_STDDEV,
        # bidPrice_ZSCORE,
        # askPrice_ZSCORE,
        # bidSize_MEAN,
        # askSize_MEAN,
        # bidSize_STDDEV,
        # askSize_STDDEV,
        # bidSize_ZSCORE,
        # askSize_ZSCORE,
        # bidPrice,
        # askPrice,
        # bidSize,
        # askSize
        #NORM_PV_1.plot()
        #plt.show()
        #exit()

        if nbSamp>0:
            ORIG_P_OS_LST_1  = self.oversample_df_OHLCv2(ORIG_P_1, tDelta, nbSamp)  # 10 mins
            NORM_PV_OS_LST_1 = self.oversample_df_OHLCv2(NORM_PV_1, tDelta, nbSamp)  # 10 mins
            for x in range(0,len(ORIG_P_OS_LST_1)):
                ORIG_P_OS_LST_1[x] = self.ZeroOutNaNs(ORIG_P_OS_LST_1[x])
                NORM_PV_OS_LST_1[x] = self.ZeroOutNaNs(NORM_PV_OS_LST_1[x])
            return ORIG_P_OS_LST_1, NORM_PV_OS_LST_1
        else:
            ORIG_P_1 = self.ZeroOutNaNs(ORIG_P_1)
            NORM_PV_1 = self.ZeroOutNaNs(NORM_PV_1)
            return ORIG_P_1, NORM_PV_1
        #print(ORIG_P_OS_LST_1)
        #print(NORM_PV_OS_LST_1)
    def loadBitmexDFFiles(self, dir, tDelta, nbSamp):
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

        ORIG_P_1 = pd.read_csv(dir + '/ORIG_P.csv', index_col=['time'], parse_dates=['time'], date_parser=dateparse)
        NORM_PV_1 = pd.read_csv(dir + '/NORM_PV.csv', index_col=['time'], parse_dates=['time'], date_parser=dateparse)

        NORM_PV_1 = NORM_PV_1[[
            'bidPrice_MEAN',
            'askPrice_MEAN',
            'bidSize_MEAN',
            'askSize_MEAN',
        ]]

        #NORM_PV_1['askPrice_ZSCORE'].plot()
        #time,
        # bidPrice_MEAN,
        # askPrice_MEAN,
        # bidPrice_STDDEV,
        # askPrice_STDDEV,
        # bidPrice_ZSCORE,
        # askPrice_ZSCORE,
        # bidSize_MEAN,
        # askSize_MEAN,
        # bidSize_STDDEV,
        # askSize_STDDEV,
        # bidSize_ZSCORE,
        # askSize_ZSCORE,
        # bidPrice,
        # askPrice,
        # bidSize,
        # askSize
        #NORM_PV_1.plot()
        #plt.show()
        #exit()

        if nbSamp>0:
            ORIG_P_OS_LST_1  = self.oversample_df_OHLCv2(ORIG_P_1, tDelta, nbSamp)  # 10 mins
            NORM_PV_OS_LST_1 = self.oversample_df_OHLCv2(NORM_PV_1, tDelta, nbSamp)  # 10 mins
            for x in range(0,len(ORIG_P_OS_LST_1)):
                ORIG_P_OS_LST_1[x] = self.ZeroOutNaNs(ORIG_P_OS_LST_1[x])
                NORM_PV_OS_LST_1[x] = self.ZeroOutNaNs(NORM_PV_OS_LST_1[x])
            return ORIG_P_OS_LST_1, NORM_PV_OS_LST_1
        else:
            ORIG_P_1 = self.ZeroOutNaNs(ORIG_P_1)
            NORM_PV_1 = self.ZeroOutNaNs(NORM_PV_1)
            return ORIG_P_1, NORM_PV_1
        #print(ORIG_P_OS_LST_1)
        #print(NORM_PV_OS_LST_1)
    def loadBitmexDF_HFT_Files(self, dir):
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')
        #2017-10-27T12:05:23.281Z
        ORIG = pd.read_csv(dir + '/OrderBookData2.txt', index_col=['TIMESTAMP'], parse_dates=['TIMESTAMP'], date_parser=dateparse)

        ORIG_P = ORIG[[
                    'BID_P1',
                    'ASK_P1'
                    ]]

        ORIG_V = ORIG[[
                    'BID_S1',
                    'BID_S2',
                    'BID_S3',
                    'BID_S4',
                    'BID_S5',
                    'BID_S6',
                    'BID_S7',
                    'BID_S8',
                    'BID_S9',
                    'ASK_S1',
                    'ASK_S2',
                    'ASK_S3',
                    'ASK_S4',
                    'ASK_S5',
                    'ASK_S6',
                    'ASK_S7',
                    'ASK_S8',
                    'ASK_S9'
                    ]]


        print(ORIG_P.head())
        print(ORIG_V.head())

        return ORIG_P,ORIG_V #NORM_PV_1
        #print(ORIG_P_OS_LST_1)
        #print(NORM_PV_OS_LST_1)

    def loadBitmexMicroStructureData(self,dir,sec,mode,lag,loadPrevDF = False):
        #mode train/test dataset
        #dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')
        # 2017-10-27T12:05:23.281Z
        #ORIG = pd.read_csv(dir + '/OrderBookData2.txt', index_col=['TIMESTAMP'], parse_dates=['TIMESTAMP'],date_parser=dateparse)

        if loadPrevDF == False:
            prevBidPrice = 0.0
            prevAskPrice = 0.0
            prevMidPrice = 0.0
            lines = []
            with open(dir + '/OrderBook.txt', 'r') as f:
                lines = f.readlines()
                #'bestMidPrice','chgMidPrice','spread'
            cols = ['time', 'chgBidPrice', 'chgAskPrice','chgMidPrice', 'bestBidPrice', 'bestAskPrice','bestMidPrice','spread', 'OBI_LVL_1', 'TSLBT', 'TSLST',
                    'BTSIZE', 'STSIZE']
            npData = np.zeros((len(lines),len(cols)))
            #dfin = pd.DataFrame()
            dfin = pd.DataFrame(data=npData, columns=cols) #dtype=np.float32

            i = 0
            for line in lines:
                #line = f.readline()
                line = line.rstrip()
                if line == '':
                    break

                json_data = ast.literal_eval(line)
                jsonLine = json.dumps(json_data)
                jsonLine = json.loads(jsonLine)

                ts = datetime.strptime(jsonLine[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")

                #CALC OBI
                OBI_LVL_1 = (float(jsonLine[0]['bids'][0][1]) - float(jsonLine[0]['asks'][0][1])) / (float(jsonLine[0]['bids'][0][1]) + float(jsonLine[0]['asks'][0][1]))
                #BEST BID<ASK PRICE
                bestBidPrice = float(jsonLine[0]['bids'][0][0])
                bestAskPrice = float(jsonLine[0]['asks'][0][0])
                bestMidPrice = (bestBidPrice + bestAskPrice)/2.0
                spread       = bestAskPrice - bestBidPrice
                #CALC CHGBID/ASKPRICE
                chgBidPrice = bestBidPrice - prevBidPrice
                chgAskPrice = bestAskPrice - prevAskPrice
                chgMidPrice = bestMidPrice - prevMidPrice
                if prevBidPrice == 0:
                    chgBidPrice = 0
                    chgAskPrice = 0
                    chgMidPrice = 0
                prevBidPrice = bestBidPrice
                prevAskPrice = bestAskPrice
                prevMidPrice = bestMidPrice
                #print(str(ts) + '\t' + str(chgBidPrice) + '\t' + str(chgAskPrice) + '\t' + str(bestBidPrice) + '\t' + str(bestAskPrice) + '\t' + str(OBI_lvl_1) )
                #pprint.pprint(jsonLine)
                #'time', 'chgBidPrice', 'chgAskPrice', 'bestBidPrice', 'bestaskPrice', 'OBI_LVL_1', 'TSLBT','TSLST', 'BTSIZE', 'STSIZE'
                dfin.set_value(i,'time',ts)
                dfin.set_value(i, 'chgBidPrice', chgBidPrice)
                dfin.set_value(i, 'chgAskPrice', chgAskPrice)
                dfin.set_value(i, 'chgMidPrice', chgMidPrice)
                dfin.set_value(i, 'bestBidPrice', bestBidPrice)
                dfin.set_value(i, 'bestAskPrice', bestAskPrice)
                dfin.set_value(i, 'bestMidPrice', bestMidPrice)
                dfin.set_value(i, 'spread', spread)
                dfin.set_value(i, 'OBI_LVL_1', OBI_LVL_1)
                dfin.set_value(i, 'TSLBT', np.nan)
                dfin.set_value(i, 'TSLST', np.nan)
                dfin.set_value(i, 'BTSIZE', np.nan)
                dfin.set_value(i, 'STSIZE', np.nan)
                #dfin.loc[i] = [ts,chgBidPrice,chgAskPrice,bestBidPrice,bestAskPrice,OBI_lvl_1,np.nan,np.nan,np.nan,np.nan]
                i = i + 1
            #construct DAtaframe
            #dfin = pd.DataFrame(columns=['time', 'chgBidPrice', 'chgAskPrice','bestBidPrice','bestaskPrice','OBI_LVL_1','TSLBT','TSLST','BTSIZE','STSIZE'])
            ##for i in range(5):
            #    df.loc[i] = [randint(-1, 1) for n in range(3)]
            print(dfin.head(10))
            print(dfin.tail(10))
            #dfin.to_csv(dir + '/MICROSTRUCTURE_DATA.csv')

            #dfin = pd.read_csv(dir + '/MICROSTRUCTURE_DATA.csv')
            #print('dfin.columns: ' + str(dfin.columns))
            #dfin = dfin[dfin.columns[:]]# the WTF column
            #print('dfin.columns: ' + str(dfin.columns))

            #dfin1 = dfin.copy()
            dfin1 = pd.DataFrame(data=dfin.as_matrix(), columns=dfin.columns, index=dfin.index)

            #dfin2 = dfin.copy()
            dfin2 = pd.DataFrame(data=dfin.as_matrix(), columns=dfin.columns, index=dfin.index)

            #dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
            #pd.read_csv(dir + '/ORIG_P_1.csv', index_col=['time'], parse_dates=['time'], date_parser=dateparse)
            #print(dfin.head(5))
            #print('OBts: ' + str(type(dfin1.get_value(0, 'time'))))

            #addTimeSinceLayButTrade_to_DF()
            df0 = self.addTimeSinceLastTrade_to_DF(dir,dfin1,'Buy')
            #df0 = df0.set_index('time')
            #print('df0.head(10): ' + str(df0[['TSLBT']].head(5)))

            df1 = self.addTimeSinceLastTrade_to_DF(dir,dfin2, 'Sell')
            #df1 = df1.set_index('time')
            #print('df1.head(10): ' + str(df1[['TSLST']].head(5)))

            print('len(df0): ' + str(len(df0)))
            print('len(df1): ' + str(len(df1)))

            '''
            for x in range(0,len(df0)):
                try:
                    ts0 = datetime.strptime(df0.get_value(x, 'time'), "%Y-%m-%d %H:%M:%S.%f")  # DF DATE FORMAT
                except:
                    ts0 = datetime.strptime(df0.get_value(x, 'time'), "%Y-%m-%d %H:%M:%S")  # DF DATE FORMAT
                try:
                    ts1 = datetime.strptime(df1.get_value(x, 'time'), "%Y-%m-%d %H:%M:%S.%f")  # DF DATE FORMAT
                except:
                    ts1 = datetime.strptime(df1.get_value(x, 'time'), "%Y-%m-%d %H:%M:%S")  # DF DATE FORMAT
    
                if ts0 == ts1:
                    print('STRANGE DATES DONT MATCH')
                    exit()
            '''
            df0[['TSLST']] = df1[['TSLST']]
            df0[['STSIZE']] = df1[['STSIZE']]
            df0 = df0.set_index('time')

            df0.to_csv(dir + '/MICROSTRUCTURE_DATA.csv')
            print('SAVE:')
            print(df0.head(10))
            DATA = []
            DATA_ACTUAL_PRICE = []
            dfTraining, dfTesting = self.splitdf(df0)
            if mode == 'TRAINING':
                DATA_ACTUAL_PRICE = dfTraining[['bestBidPrice','bestAskPrice']]
                DATA = dfTraining[['chgBidPrice','chgAskPrice','OBI_LVL_1','TSLBT','TSLST']]
            elif mode == 'TESTING':
                DATA_ACTUAL_PRICE = dfTesting[['bestBidPrice', 'bestAskPrice']]
                DATA = dfTesting[['chgBidPrice', 'chgAskPrice', 'OBI_LVL_1', 'TSLBT', 'TSLST']]
            elif mode == 'TRAINING_V2':
                print("ADDED BTSIZE','STSIZE'")
                DATA_ACTUAL_PRICE = dfTraining[['bestBidPrice', 'bestAskPrice']]
                DATA = dfTraining[['chgBidPrice', 'chgAskPrice', 'OBI_LVL_1', 'TSLBT', 'TSLST','BTSIZE','STSIZE']]
            elif mode == 'TESTING_V2':
                print("ADDED BTSIZE','STSIZE'")
                DATA_ACTUAL_PRICE = dfTesting[['bestBidPrice', 'bestAskPrice']]
                DATA = dfTesting[['chgBidPrice', 'chgAskPrice', 'OBI_LVL_1', 'TSLBT', 'TSLST','BTSIZE','STSIZE']]
            elif mode == 'TRAINING_V3':
                #dfTraining, dfTesting = self.splitdf(df0)
                # dfTraining, dfTesting = self.splitdf(dfTraining)
                # dfTraining, dfTesting = self.splitdf(dfTraining) #7hrs
                # dfTraining, dfTesting = self.splitdf(dfTraining) #4
                dfTraining3, dfTesting4 = self.splitdf(dfTraining)  # 2

                part_3 = lambda: self.conditionData(dfTraining3, dfTesting, mode, sec, dir, '.part3',lag)
                part_4 = lambda: self.conditionData(dfTesting4, dfTesting, mode, sec, dir, '.part4',lag)


                thread3 = threading.Thread(target=(part_3))
                thread3.daemon = True
                thread3.start()

                thread4 = threading.Thread(target=(part_4))
                thread4.daemon = True
                thread4.start()

                while thread3.is_alive() == True or \
                                thread4.is_alive() == True:
                    print ('still alive')
                    time.sleep(5)

                # time.sleep(120)
                print('fin')
                # exit()
                dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

                DATA_3 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1.csv' + '.part3', index_col=['time'],
                                     parse_dates=['time'], date_parser=dateparse)
                DATA_4 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1.csv' + '.part4', index_col=['time'],
                                     parse_dates=['time'], date_parser=dateparse)

                DATA_ACTUAL_PRICE_3 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2.csv' + '.part3', index_col=['time'],
                                                  parse_dates=['time'], date_parser=dateparse)
                DATA_ACTUAL_PRICE_4 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2.csv' + '.part4', index_col=['time'],
                                                  parse_dates=['time'], date_parser=dateparse)

                df_lst_ACTUAL = []
                df_lst_ACTUAL.append(DATA_ACTUAL_PRICE_3)
                df_lst_ACTUAL.append(DATA_ACTUAL_PRICE_4)

                # print('DATA_1: ' + str(DATA_1))
                df_lst = []
                df_lst.append(DATA_3)
                df_lst.append(DATA_4)

                DATA = np.concatenate([DATA_3.as_matrix(), DATA_4.as_matrix()], axis=0)

                DATA_ACTUAL = np.concatenate([DATA_ACTUAL_PRICE_3.as_matrix(), DATA_ACTUAL_PRICE_4.as_matrix()], axis=0)

                indexes = []  # np.zeros((len(DATA)),dtype=float)
                indexes_ACTUAL = []
                for df in df_lst:
                    for ind in df.index:
                        indexes.append(ind)

                for df in df_lst_ACTUAL:
                    for ind in df.index:
                        indexes_ACTUAL.append(ind)

                print('DATA: ' + str(DATA))
                df_DATA = pd.DataFrame(data=DATA, columns=DATA_3.columns, index=indexes, dtype=np.float32)
                df_DATA_ACTUAL = pd.DataFrame(data=DATA_ACTUAL, columns=DATA_ACTUAL_PRICE_3.columns,
                                              index=indexes_ACTUAL, dtype=np.float32)

                print('df_DATA: ' + str(df_DATA))
                print('df_DATA_ACTUAL: ' + str(df_DATA_ACTUAL))

                # df_DATA_ACTUAL[['exitSL_PriceLong','bestAskPrice']].plot() ##bestAskPrice bestBidPrice  pcentPriceChgLong
                # pcentPriceChgShort  exitSL_PriceLong  exitSL_PriceShort
                # entryPriceValue  exitPriceValueLong  exitPriceValueShort
                # plt.show()
                # plt.hold(True)
                # print('DATA: ' + str(DATA))

                # exit()
                # self.conditionData(dfTraining,dfTesting,mode,sec,dir,'.part1')

                return df_DATA, df_DATA_ACTUAL
            elif mode == 'TESTING_V3':
                # dfTraining, dfTesting = self.splitdf(df0)
                # dfTraining, dfTesting = self.splitdf(dfTraining)
                # dfTraining, dfTesting = self.splitdf(dfTraining) #7hrs
                # dfTraining, dfTesting = self.splitdf(dfTraining) #4
                #dfTraining1, dfTesting2 = self.splitdf(dfTesting)  # 2
                dfTraining3, dfTesting4 = self.splitdf(dfTesting)  # 2

                part_3 = lambda: self.conditionData(dfTraining3, dfTesting, mode, sec, dir, '.part3', lag)
                part_4 = lambda: self.conditionData(dfTesting4, dfTesting, mode, sec, dir, '.part4', lag)


                thread3 = threading.Thread(target=(part_3))
                thread3.daemon = True
                thread3.start()

                thread4 = threading.Thread(target=(part_4))
                thread4.daemon = True
                thread4.start()

                while thread3.is_alive() == True or thread4.is_alive() == True:
                    print ('still alive')
                    time.sleep(5)

                # time.sleep(120)
                print('fin')
                # exit()
                dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

                DATA_3 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1.csv' + '.part3', index_col=['time'],
                                     parse_dates=['time'], date_parser=dateparse)
                DATA_4 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1.csv' + '.part4', index_col=['time'],
                                     parse_dates=['time'], date_parser=dateparse)

                DATA_ACTUAL_PRICE_3 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2.csv' + '.part3', index_col=['time'],
                                                  parse_dates=['time'], date_parser=dateparse)
                DATA_ACTUAL_PRICE_4 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2.csv' + '.part4', index_col=['time'],
                                                  parse_dates=['time'], date_parser=dateparse)

                df_lst_ACTUAL = []
                df_lst_ACTUAL.append(DATA_ACTUAL_PRICE_3)
                df_lst_ACTUAL.append(DATA_ACTUAL_PRICE_4)

                # print('DATA_1: ' + str(DATA_1))
                df_lst = []
                df_lst.append(DATA_3)
                df_lst.append(DATA_4)

                DATA = np.concatenate([DATA, DATA_3.as_matrix()], axis=0)
                DATA = np.concatenate([DATA, DATA_4.as_matrix()], axis=0)

                DATA_ACTUAL = np.concatenate([DATA_ACTUAL_PRICE_3.as_matrix(), DATA_ACTUAL_PRICE_4.as_matrix()], axis=0)

                indexes = []  # np.zeros((len(DATA)),dtype=float)
                indexes_ACTUAL = []
                for df in df_lst:
                    for ind in df.index:
                        indexes.append(ind)

                for df in df_lst_ACTUAL:
                    for ind in df.index:
                        indexes_ACTUAL.append(ind)

                print('DATA: ' + str(DATA))
                df_DATA = pd.DataFrame(data=DATA, columns=DATA_3.columns, index=indexes, dtype=np.float32)
                df_DATA_ACTUAL = pd.DataFrame(data=DATA_ACTUAL, columns=DATA_ACTUAL_PRICE_3.columns,
                                              index=indexes_ACTUAL, dtype=np.float32)

                print('df_DATA: ' + str(df_DATA))
                print('df_DATA_ACTUAL: ' + str(df_DATA_ACTUAL))

                # df_DATA_ACTUAL[['exitSL_PriceLong','bestAskPrice']].plot() ##bestAskPrice bestBidPrice  pcentPriceChgLong
                # pcentPriceChgShort  exitSL_PriceLong  exitSL_PriceShort
                # entryPriceValue  exitPriceValueLong  exitPriceValueShort
                # plt.show()
                # plt.hold(True)
                # print('DATA: ' + str(DATA))

                # exit()
                # self.conditionData(dfTraining,dfTesting,mode,sec,dir,'.part1')

                return df_DATA, df_DATA_ACTUAL

        else:

            dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

            DATA_1 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1.csv' + '.part1', index_col=['time'], parse_dates=['time'],date_parser=dateparse)
            DATA_2 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1.csv' + '.part2', index_col=['time'], parse_dates=['time'],date_parser=dateparse)
            DATA_3 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1.csv' + '.part3', index_col=['time'], parse_dates=['time'],date_parser=dateparse)
            DATA_4 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1.csv' + '.part4', index_col=['time'], parse_dates=['time'],date_parser=dateparse)

            DATA_ACTUAL_PRICE_1 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2.csv' + '.part1', index_col=['time'], parse_dates=['time'], date_parser=dateparse)
            DATA_ACTUAL_PRICE_2 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2.csv' + '.part2', index_col=['time'], parse_dates=['time'], date_parser=dateparse)
            DATA_ACTUAL_PRICE_3 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2.csv' + '.part3', index_col=['time'], parse_dates=['time'], date_parser=dateparse)
            DATA_ACTUAL_PRICE_4 = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2.csv' + '.part4', index_col=['time'], parse_dates=['time'], date_parser=dateparse)

            df_lst_ACTUAL = []
            df_lst_ACTUAL.append(DATA_ACTUAL_PRICE_1)
            df_lst_ACTUAL.append(DATA_ACTUAL_PRICE_2)
            df_lst_ACTUAL.append(DATA_ACTUAL_PRICE_3)
            df_lst_ACTUAL.append(DATA_ACTUAL_PRICE_4)

            #print('DATA_1: ' + str(DATA_1))
            df_lst = []
            df_lst.append(DATA_1)
            df_lst.append(DATA_2)
            df_lst.append(DATA_3)
            df_lst.append(DATA_4)

            DATA = np.concatenate([DATA_1.as_matrix(), DATA_2.as_matrix()], axis=0)
            DATA = np.concatenate([DATA, DATA_3.as_matrix()], axis=0)
            DATA = np.concatenate([DATA, DATA_4.as_matrix()], axis=0)

            DATA_ACTUAL = np.concatenate([DATA_ACTUAL_PRICE_1.as_matrix(), DATA_ACTUAL_PRICE_2.as_matrix()], axis=0)
            DATA_ACTUAL = np.concatenate([DATA_ACTUAL, DATA_ACTUAL_PRICE_3.as_matrix()], axis=0)
            DATA_ACTUAL = np.concatenate([DATA_ACTUAL, DATA_ACTUAL_PRICE_4.as_matrix()], axis=0)

            indexes = []#np.zeros((len(DATA)),dtype=float)
            indexes_ACTUAL = []
            for df in df_lst:
                for ind in df.index:
                    indexes.append(ind)

            for df in df_lst_ACTUAL:
                for ind in df.index:
                    indexes_ACTUAL.append(ind)

            print('DATA: ' + str(DATA))
            df_DATA = pd.DataFrame(data=DATA, columns=DATA_1.columns, index=indexes, dtype=np.float32)
            df_DATA_ACTUAL = pd.DataFrame(data=DATA_ACTUAL, columns=DATA_ACTUAL_PRICE_1.columns, index=indexes_ACTUAL, dtype=np.float32)

            print('df_DATA: ' + str(df_DATA))
            print('df_DATA_ACTUAL: ' + str(df_DATA_ACTUAL))

            #df_DATA_ACTUAL[['exitSL_PriceLong','bestAskPrice']].plot() ##bestAskPrice bestBidPrice  pcentPriceChgLong
            #pcentPriceChgShort  exitSL_PriceLong  exitSL_PriceShort
            #entryPriceValue  exitPriceValueLong  exitPriceValueShort
            #plt.show()
            #plt.hold(True)
            #print('DATA: ' + str(DATA))

            #exit()
            #self.conditionData(dfTraining,dfTesting,mode,sec,dir,'.part1')

            return df_DATA,df_DATA_ACTUAL

    def loadBitmexMicroStructureDataV5(self, dir, sec, lag,mode, loadPrevDF=False):
        # mode train/test dataset
        # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')
        # 2017-10-27T12:05:23.281Z
        # ORIG = pd.read_csv(dir + '/OrderBookData2.txt', index_col=['TIMESTAMP'], parse_dates=['TIMESTAMP'],date_parser=dateparse)
        #lag = 10
        if loadPrevDF == False:
            prevBidPrice = 0.0
            prevAskPrice = 0.0
            prevMidPrice = 0.0
            lines = []
            with open(dir + '/OrderBook.txt', 'r') as f:
                lines = f.readlines()
            cols = ['time', 'chgBidP', 'chgAskP', 'chgMidP', 'bidP', 'askP', 'midP',
                    'spread', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5',
                    'bidS_1', 'askS_1']
            #df = pd.DataFrame(columns=cols)  # dtype=np.float32
            npData = np.zeros((len(lines), len(cols)))
            # dfin = pd.DataFrame()
            df = pd.DataFrame(data=npData, columns=cols)  # dtype=np.float32

            i = 0
            for line in lines:
                line = line.rstrip()
                if line == '':
                    break

                json_data = ast.literal_eval(line)
                jsonLine = json.dumps(json_data)
                jsonLine = json.loads(jsonLine)

                ts = datetime.strptime(jsonLine[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")

                # CALC OBI
                OBI_LVL_1 = (float(jsonLine[0]['bids'][0][1]) - float(jsonLine[0]['asks'][0][1])) / (float(jsonLine[0]['bids'][0][1]) + float(jsonLine[0]['asks'][0][1]))
                OBI_LVL_2 = (float(jsonLine[0]['bids'][1][1]) - float(jsonLine[0]['asks'][1][1])) / (float(jsonLine[0]['bids'][1][1]) + float(jsonLine[0]['asks'][1][1]))
                OBI_LVL_3 = (float(jsonLine[0]['bids'][2][1]) - float(jsonLine[0]['asks'][2][1])) / (float(jsonLine[0]['bids'][2][1]) + float(jsonLine[0]['asks'][2][1]))
                OBI_LVL_4 = (float(jsonLine[0]['bids'][3][1]) - float(jsonLine[0]['asks'][3][1])) / (float(jsonLine[0]['bids'][3][1]) + float(jsonLine[0]['asks'][3][1]))
                OBI_LVL_5 = (float(jsonLine[0]['bids'][4][1]) - float(jsonLine[0]['asks'][4][1])) / (float(jsonLine[0]['bids'][4][1]) + float(jsonLine[0]['asks'][4][1]))
                bidS_1 = float(jsonLine[0]['bids'][0][1])
                askS_1 = float(jsonLine[0]['asks'][0][1])
                # BEST BID<ASK PRICE
                bidP = float(jsonLine[0]['bids'][0][0])
                askP = float(jsonLine[0]['asks'][0][0])
                bestMidPrice = (bidP + askP) / 2.0
                spread = askP - bidP
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

                #['time', 'chgBidP', 'chgAskP', 'chgMidP', 'bidP', 'askP', 'midP',
                #'spread', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5',
                #'bidS_1', 'askS_1']
                #newRow = [ts, chgBidPrice, chgAskPrice, chgMidPrice, bidP, askP, bestMidPrice, spread, OBI_LVL_1,
                #          OBI_LVL_2, OBI_LVL_3, OBI_LVL_4, OBI_LVL_5, bidS_1, askS_1]
                #df = df.append(pd.Series(newRow, index=cols), ignore_index=True)
                df.set_value(i, 'time', ts)
                df.set_value(i, 'chgBidP', chgBidPrice)
                df.set_value(i, 'chgAskP', chgAskPrice)
                df.set_value(i, 'chgMidP', chgMidPrice)
                df.set_value(i, 'bidP', bidP)
                df.set_value(i, 'askP', askP)
                df.set_value(i, 'midP', bestMidPrice)
                df.set_value(i, 'spread', spread)
                df.set_value(i, 'OBI_LVL_1', OBI_LVL_1)
                df.set_value(i, 'OBI_LVL_2', OBI_LVL_2)
                df.set_value(i, 'OBI_LVL_3', OBI_LVL_3)
                df.set_value(i, 'OBI_LVL_4', OBI_LVL_4)
                df.set_value(i, 'OBI_LVL_5', OBI_LVL_5)
                df.set_value(i, 'bidS_1', bidS_1)
                df.set_value(i, 'askS_1', askS_1)

                i = i + 1
                print (str(i))
            df = df.set_index('time')
            df.to_csv(dir + '/MICROSTRUCTURE_DATA.csv')
            dfTraining, dfTesting = self.splitdf(df) #6days
            #dfTraining #3days
            dfTraining, dfTesting = self.splitdf(dfTraining)  # 6days
            #dfTraining  # 1.5days
            dfTraining, dfTesting = self.splitdf(dfTraining)  # 6days
            #dfTraining  # 0.75days
            if mode =='testing':
                DATA_ACTUAL_PRICE = dfTesting[['bidP', 'askP']]
                DATA = dfTesting[
                ['spread', 'chgMidP', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5', 'bidS_1',
                 'askS_1']]

            else:
                DATA = dfTraining[['spread', 'chgMidP', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5','bidS_1', 'askS_1']]
                DATA_ACTUAL_PRICE = dfTraining[['bidP', 'askP']]


            DATA_ACTUAL_PRICE = DATA_ACTUAL_PRICE.resample(str(sec) + 'S',
                                                           how={'askP': 'mean', 'bidP': 'mean'}).bfill()

            DATA = DATA.resample(str(sec) + 'S',
                                 how={'spread': 'max', 'chgMidP': 'sum', 'OBI_LVL_1': 'mean', 'OBI_LVL_2': 'mean',
                                      'OBI_LVL_3': 'mean', 'OBI_LVL_4': 'mean', 'OBI_LVL_5': 'mean', 'bidS_1': 'sum', 'askS_1': 'sum'}).bfill()

            DATA_ACTUAL_PRICE['pcentPriceChgLong'] = DATA['chgMidP']
            DATA_ACTUAL_PRICE['pcentPriceChgShort'] = DATA['chgMidP']
            DATA_ACTUAL_PRICE['entryPriceLong'] = DATA['chgMidP']
            DATA_ACTUAL_PRICE['entryPriceShort'] = DATA['chgMidP']
            DATA_ACTUAL_PRICE['exitPriceLong'] = DATA['chgMidP']
            DATA_ACTUAL_PRICE['exitPriceShort'] = DATA['chgMidP']
            DATA_ACTUAL_PRICE['entryIndex'] = DATA['chgMidP']
            DATA_ACTUAL_PRICE['exitIndexLong'] = DATA['chgMidP']
            DATA_ACTUAL_PRICE['exitIndexShort'] = DATA['chgMidP']

            pcentPriceChgLong = 0
            pcentPriceChgShort = 0
            exitPriceValueLong = 0
            exitPriceValueShort = 0
            print ('HERE')
            print('perfect exit strategy')
            if lag > len(DATA):
                print('lag value too large: ' + str(lag))
                exit()
            for x in range(0, len(DATA) - lag):
                if x % 500 == 0:
                    print(str(Thread.name) + ': ' + str(x) + ' of ' + str(len(DATA)))
                entryPriceLong = float(DATA_ACTUAL_PRICE['askP'][x])
                entryPriceShort = float(DATA_ACTUAL_PRICE['bidP'][x])
                indexes = DATA_ACTUAL_PRICE.index
                entryIndex = x  # indexes[x]

                maxExitPriceLong = -9999999
                minExitPriceShort = 9999999
                npBestBidPrices = DATA_ACTUAL_PRICE['bidP'].as_matrix()
                npBestAskPrices = DATA_ACTUAL_PRICE['askP'].as_matrix()
                ExitIndexLong = ''
                ExitIndexShort = ''

                for n in range(x, x + lag):
                    exitPriceLong = npBestBidPrices[n]
                    if maxExitPriceLong < exitPriceLong:
                        # new max
                        ExitIndexLong = n  # indexes[n]
                        maxExitPriceLong = exitPriceLong

                    exitPriceShort = npBestAskPrices[n]
                    if minExitPriceShort > exitPriceShort:
                        # new max
                        ExitIndexShort = n  # indexes[n]
                        minExitPriceShort = exitPriceShort

                pcentPriceChgLong = ((maxExitPriceLong - entryPriceLong) / entryPriceLong) * 100
                pcentPriceChgShort = ((minExitPriceShort - entryPriceShort) / entryPriceShort) * 100

                if (-1 * pcentPriceChgShort) >= pcentPriceChgLong:
                    pcentPriceChgLong = 0.0
                    #pcentPriceChgShort = -0.6
                else:
                    pcentPriceChgShort = 0.0
                    #pcentPriceChgLong = 0.6

                DATA_ACTUAL_PRICE['pcentPriceChgLong'][x] = pcentPriceChgLong
                DATA_ACTUAL_PRICE['pcentPriceChgShort'][x] = pcentPriceChgShort
                DATA_ACTUAL_PRICE['entryPriceLong'][x] = entryPriceLong
                DATA_ACTUAL_PRICE['entryPriceShort'][x] = entryPriceShort
                DATA_ACTUAL_PRICE['exitPriceLong'][x] = maxExitPriceLong
                DATA_ACTUAL_PRICE['exitPriceShort'][x] = minExitPriceShort
                DATA_ACTUAL_PRICE['entryIndex'][x] = entryIndex
                DATA_ACTUAL_PRICE['exitIndexLong'][x] = ExitIndexLong
                DATA_ACTUAL_PRICE['exitIndexShort'][x] = ExitIndexShort

            for x in range(len(DATA) - lag, len(DATA)):
                #print(str(x))
                entryPriceLong = float(DATA_ACTUAL_PRICE['askP'][x])
                entryPriceShort = float(DATA_ACTUAL_PRICE['bidP'][x])
                indexes = DATA_ACTUAL_PRICE.index
                entryIndex = x  # indexes[x]

                DATA_ACTUAL_PRICE['pcentPriceChgLong'][x] = 0.0
                DATA_ACTUAL_PRICE['pcentPriceChgShort'][x] = 0.0
                DATA_ACTUAL_PRICE['entryPriceLong'][x] = entryPriceLong
                DATA_ACTUAL_PRICE['entryPriceShort'][x] = entryPriceShort
                DATA_ACTUAL_PRICE['exitPriceLong'][x] = entryPriceLong
                DATA_ACTUAL_PRICE['exitPriceShort'][x] = entryPriceShort
                DATA_ACTUAL_PRICE['entryIndex'][x] = entryIndex
                DATA_ACTUAL_PRICE['exitIndexLong'][x] = entryIndex
                DATA_ACTUAL_PRICE['exitIndexShort'][x] = entryIndex
            if mode == 'testing':
                DATA.to_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1_TEST.csv',index=False)
                DATA_ACTUAL_PRICE.to_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2_TEST.csv',index=False)
            else:
                DATA.to_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1_TRAIN.csv',index=False)
                DATA_ACTUAL_PRICE.to_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2_TRAIN.csv',index=False)

            #print ('AASDF')
            return DATA, DATA_ACTUAL_PRICE
        else:
            dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

            if mode == 'testing':
                DATA = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1_TEST.csv', index_col=['time'], parse_dates=['time'],
                                   date_parser=dateparse)
                DATA_ACTUAL_PRICE = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2_TEST.csv', index_col=['time'],
                                                parse_dates=['time'], date_parser=dateparse)
            else:
                DATA = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1_TRAIN.csv', index_col=['time'], parse_dates=['time'],
                                   date_parser=dateparse)
                DATA_ACTUAL_PRICE = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2_TRAIN.csv', index_col=['time'],
                                                parse_dates=['time'], date_parser=dateparse)


            return DATA, DATA_ACTUAL_PRICE

    def collectDatav3(self, dir, lag, mode,tDelay, loadPrevDF=False):

        if loadPrevDF == False:
            prevBidPrice = 0.0
            prevAskPrice = 0.0
            prevMidPrice = 0.0
            spread_lst=[]
            chgMidPrice_lst=[]
            OBI_LVL_1_lst=[]
            OBI_LVL_2_lst=[]
            OBI_LVL_3_lst=[]
            OBI_LVL_4_lst=[]
            OBI_LVL_5_lst=[]
            bidS_1_lst=[]
            askS_1_lst=[]
            bidP_1_lst = []
            askP_1_lst = []
            data_lst = []
            data_actual_lst = []
            with open(dir + '/OrderBook.txt', 'r') as f:
                lines = f.readlines()
            if mode == 'training':
                lines = lines[0:int(len(lines)/2)]
            else:
                lines = lines[int(len(lines) / 2):]

            line = lines[0].rstrip()

            json_data = ast.literal_eval(line)
            jsonLine = json.dumps(json_data)
            jsonLine = json.loads(jsonLine)
            curr_ts = datetime.strptime(jsonLine[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
            cols = ['spread', 'chgMidP', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5', 'bidS_1','askS_1']
            #df = pd.DataFrame(columns=cols)
            #df_actual = pd.DataFrame(columns=['bidP', 'askP'])

            i = 0
            for line in lines:
                line = line.rstrip()
                if line == '':
                    break

                json_data = ast.literal_eval(line)
                jsonLine = json.dumps(json_data)
                jsonLine = json.loads(jsonLine)

                ts = datetime.strptime(jsonLine[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")

                bidS_1 = float(jsonLine[0]['bids'][0][1])
                askS_1 = float(jsonLine[0]['asks'][0][1])
                bidS_2 = float(jsonLine[0]['bids'][1][1])
                askS_2 = float(jsonLine[0]['asks'][1][1])
                bidS_3 = float(jsonLine[0]['bids'][2][1])
                askS_3 = float(jsonLine[0]['asks'][2][1])
                bidS_4 = float(jsonLine[0]['bids'][3][1])
                askS_4 = float(jsonLine[0]['asks'][3][1])
                bidS_5 = float(jsonLine[0]['bids'][4][1])
                askS_5 = float(jsonLine[0]['asks'][4][1])

                # CALC OBI
                OBI_LVL_1 = (bidS_1 - askS_1) / (bidS_1 + askS_1)
                OBI_LVL_2 = (bidS_2 - askS_2) / (bidS_2 + askS_2)
                OBI_LVL_3 = (bidS_3 - askS_3) / (bidS_3 + askS_3)
                OBI_LVL_4 = (bidS_4 - askS_4) / (bidS_4 + askS_4)
                OBI_LVL_5 = (bidS_5 - askS_5) / (bidS_5 + askS_5)


                # BEST BID<ASK PRICE
                bidP = float(jsonLine[0]['bids'][0][0])
                askP = float(jsonLine[0]['asks'][0][0])
                bestMidPrice = (bidP + askP) / 2.0
                spread = askP - bidP
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


                #print (str(chgMidPrice))
                #print (str(bestMidPrice))
                #print (str(prevMidPrice))
                spread_lst.append(spread)
                chgMidPrice_lst.append(chgMidPrice)
                OBI_LVL_1_lst.append(OBI_LVL_1)
                OBI_LVL_2_lst.append(OBI_LVL_2)
                OBI_LVL_3_lst.append(OBI_LVL_3)
                OBI_LVL_4_lst.append(OBI_LVL_4)
                OBI_LVL_5_lst.append(OBI_LVL_5)
                bidS_1_lst.append(bidS_1)
                askS_1_lst.append(askS_1)
                bidP_1_lst.append(bidP)
                askP_1_lst.append(askP)

                if (curr_ts + timedelta(0, tDelay)) < ts and spread_lst != []:
                    curr_ts = datetime.strptime(jsonLine[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")

                    spread_New = np.asarray(spread_lst).mean()
                    chgMidPrice_New = np.asarray(chgMidPrice_lst).cumsum()[-1]
                    OBI_LVL_1_New = np.asarray(OBI_LVL_1_lst).mean()
                    OBI_LVL_2_New = np.asarray(OBI_LVL_2_lst).mean()
                    OBI_LVL_3_New = np.asarray(OBI_LVL_3_lst).mean()
                    OBI_LVL_4_New = np.asarray(OBI_LVL_4_lst).mean()
                    OBI_LVL_5_New = np.asarray(OBI_LVL_5_lst).mean()
                    bidS_1_New = np.asarray(bidS_1_lst).mean()
                    askS_1_New = np.asarray(askS_1_lst).mean()#.mean()
                    bidP_1_New = np.asarray(bidP_1_lst).mean()#.mean()
                    askP_1_New = np.asarray(askP_1_lst).mean()#.mean()

                    data = [spread_New,
                             chgMidPrice_New,
                             OBI_LVL_1_New,
                             OBI_LVL_2_New,
                             OBI_LVL_3_New,
                             OBI_LVL_4_New,
                             OBI_LVL_5_New,
                             bidS_1_New,
                             askS_1_New]

                    data_actual = [bidP_1_New,askP_1_New]

                    data_lst.append(data)
                    data_actual_lst.append(data_actual)
                    #df = df.append(data,ignore_index=True)
                    #df_actual = df_actual.append(data_actual,ignore_index=True)
                    #df.dropna(inplace=True)

                    spread_lst = []
                    chgMidPrice_lst = []
                    OBI_LVL_1_lst = []
                    OBI_LVL_2_lst = []
                    OBI_LVL_3_lst = []
                    OBI_LVL_4_lst = []
                    OBI_LVL_5_lst = []
                    bidS_1_lst = []
                    askS_1_lst = []
                    bidP_1_lst = []
                    askP_1_lst = []

                i = i + 1
                if i != 0 and i % 10000 == 0:
                    print (str(i))

            npData = np.asarray(data_lst)
            npDataActual = np.asarray(data_actual_lst)
            df = pd.DataFrame(data=npData,columns=cols)
            df_actual = pd.DataFrame(data=npDataActual,columns=['bidP', 'askP'])

            print('df: ' + str(df))
            print('df_actual: ' + str(df_actual))

            DATA, DATA_ACTUAL_PRICE = self.calcPerfectEntryStrat(df,df_actual,lag,mode,dir)
            #DATA = df
            #DATA_ACTUAL_PRICE = df_actual

            return DATA, DATA_ACTUAL_PRICE
        else:
            #dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

            if mode == 'testing':
                DATA = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1_TEST_V2.csv')
                DATA_ACTUAL_PRICE = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2_TEST_V2.csv')
                DATA.set_index('count')
                DATA_ACTUAL_PRICE.set_index('count')
                DATA = DATA.drop('count', 1)

                return DATA, DATA_ACTUAL_PRICE
            else:
                DATA = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1_TRAIN_V2.csv')
                DATA_ACTUAL_PRICE = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2_TRAIN_V2.csv')

                DATA.set_index('count')
                DATA_ACTUAL_PRICE.set_index('count')
                DATA = DATA.drop('count', 1)

                print('DATA.columns: ' +str(DATA.columns))
                print('DATA_ACTUAL_PRICE.columns: ' + str(DATA_ACTUAL_PRICE.columns))


                return DATA, DATA_ACTUAL_PRICE

    def calcPerfectEntryStrat(self,DATA,DATA_ACTUAL_PRICE,lag,mode,dir):
        DATA_ACTUAL_PRICE['pcentPriceChgLong'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['pcentPriceChgShort'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['entryPriceLong'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['entryPriceShort'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['exitPriceLong'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['exitPriceShort'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['entryIndex'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['exitIndexLong'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['exitIndexShort'] = DATA['chgMidP']

        pcentPriceChgLong = 0
        pcentPriceChgShort = 0
        exitPriceValueLong = 0
        exitPriceValueShort = 0
        print ('HERE')
        print('perfect exit strategy')
        if lag > len(DATA):
            print('lag value too large: ' + str(lag))
            exit()
        for x in range(0, len(DATA) - lag):
            if x % 500 == 0:
                print(str(Thread.name) + ': ' + str(x) + ' of ' + str(len(DATA)))
            entryPriceLong = float(DATA_ACTUAL_PRICE['askP'][x])
            entryPriceShort = float(DATA_ACTUAL_PRICE['bidP'][x])
            indexes = DATA_ACTUAL_PRICE.index
            entryIndex = x  # indexes[x]

            maxExitPriceLong = -9999999
            minExitPriceShort = 9999999
            npBestBidPrices = DATA_ACTUAL_PRICE['bidP'].as_matrix()
            npBestAskPrices = DATA_ACTUAL_PRICE['askP'].as_matrix()
            ExitIndexLong = ''
            ExitIndexShort = ''

            for n in range(x, x + lag):
                exitPriceLong = npBestBidPrices[n]
                if maxExitPriceLong < exitPriceLong:
                    # new max
                    ExitIndexLong = n  # indexes[n]
                    maxExitPriceLong = exitPriceLong

                exitPriceShort = npBestAskPrices[n]
                if minExitPriceShort > exitPriceShort:
                    # new max
                    ExitIndexShort = n  # indexes[n]
                    minExitPriceShort = exitPriceShort

            pcentPriceChgLong = ((maxExitPriceLong - entryPriceLong) / entryPriceLong) * 100
            pcentPriceChgShort = ((minExitPriceShort - entryPriceShort) / entryPriceShort) * 100

            if (-1 * pcentPriceChgShort) >= pcentPriceChgLong:
                pcentPriceChgLong = 0.0
                # pcentPriceChgShort = -0.6
            else:
                pcentPriceChgShort = 0.0
                # pcentPriceChgLong = 0.6

            DATA_ACTUAL_PRICE['pcentPriceChgLong'][x] = pcentPriceChgLong
            DATA_ACTUAL_PRICE['pcentPriceChgShort'][x] = pcentPriceChgShort
            DATA_ACTUAL_PRICE['entryPriceLong'][x] = entryPriceLong
            DATA_ACTUAL_PRICE['entryPriceShort'][x] = entryPriceShort
            DATA_ACTUAL_PRICE['exitPriceLong'][x] = maxExitPriceLong
            DATA_ACTUAL_PRICE['exitPriceShort'][x] = minExitPriceShort
            DATA_ACTUAL_PRICE['entryIndex'][x] = entryIndex
            DATA_ACTUAL_PRICE['exitIndexLong'][x] = ExitIndexLong
            DATA_ACTUAL_PRICE['exitIndexShort'][x] = ExitIndexShort

        for x in range(len(DATA) - lag, len(DATA)):
            # print(str(x))
            entryPriceLong = float(DATA_ACTUAL_PRICE['askP'][x])
            entryPriceShort = float(DATA_ACTUAL_PRICE['bidP'][x])
            indexes = DATA_ACTUAL_PRICE.index
            entryIndex = x  # indexes[x]

            DATA_ACTUAL_PRICE['pcentPriceChgLong'][x] = 0.0
            DATA_ACTUAL_PRICE['pcentPriceChgShort'][x] = 0.0
            DATA_ACTUAL_PRICE['entryPriceLong'][x] = entryPriceLong
            DATA_ACTUAL_PRICE['entryPriceShort'][x] = entryPriceShort
            DATA_ACTUAL_PRICE['exitPriceLong'][x] = entryPriceLong
            DATA_ACTUAL_PRICE['exitPriceShort'][x] = entryPriceShort
            DATA_ACTUAL_PRICE['entryIndex'][x] = entryIndex
            DATA_ACTUAL_PRICE['exitIndexLong'][x] = entryIndex
            DATA_ACTUAL_PRICE['exitIndexShort'][x] = entryIndex

            #DATA_Training, DATA_Testing = self.splitdf(DATA)  # 6days
            #DATA_ACTUAL_PRICE_Training, DATA_ACTUAL_PRICE_Testing = self.splitdf(DATA_ACTUAL_PRICE)  # 6days

        if mode == 'testing':
            DATA.to_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1_TEST_V2.csv',index_label='count')
            DATA_ACTUAL_PRICE.to_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2_TEST_V2.csv',index_label='count')
            DATA = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1_TEST_V2.csv')
            DATA_ACTUAL_PRICE = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2_TEST_V2.csv')
            DATA.set_index('count')
            DATA_ACTUAL_PRICE.set_index('count')
            DATA = DATA.drop('count', 1)

            return DATA, DATA_ACTUAL_PRICE
        else:
            DATA.to_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1_TRAIN_V2.csv',index_label='count')
            DATA_ACTUAL_PRICE.to_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2_TRAIN_V2.csv',index_label='count')
            DATA = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_1_TRAIN_V2.csv')
            DATA_ACTUAL_PRICE = pd.read_csv(dir + '/MICROSTRUCTURE_DATA_OUT_2_TRAIN_V2.csv')

            DATA.set_index('count')
            DATA_ACTUAL_PRICE.set_index('count')
            DATA = DATA.drop('count', 1)

            return DATA, DATA_ACTUAL_PRICE

    def loadBitmexMicroStructureDataV4(self, dir,sec,mode,loadPrevDF = False):
        # mode train/test dataset
        # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')
        # 2017-10-27T12:05:23.281Z
        # ORIG = pd.read_csv(dir + '/OrderBookData2.txt', index_col=['TIMESTAMP'], parse_dates=['TIMESTAMP'],date_parser=dateparse)
        #worker_threads = []
        if loadPrevDF == False:
            df_OB = self.convertRawOrderBookToDF(dir) #['time','spread','BidPrice_1', 'AskPrice_1', 'MidPrice_1','BidSize_1', 'AskSize_1']
            df_TR = self.convertRawLatestTradesToDF(dir) #['time','side','price', 'sizeInBTC', 'sizeInUSD','nbOfTrades']
            '''
            worker1 = lambda: self.convertRawOrderBookToDF(dir)
            worker2 = lambda: self.convertRawLatestTradesToDF(dir)
    
            thread1 = threading.Thread(target=(worker1))
            thread2 = threading.Thread(target=(worker2))
            thread1.daemon = True
            thread2.daemon = True
            thread1.start()
            thread2.start()
    
            while thread1.is_alive or thread2.is_alive:
                pass
            
            print('threads fin')
            df_OB = self.df_OB
            df_TR = self.df_TR
            '''
            df_OB = df_OB.set_index('time')
            df_TR = df_TR.set_index('time')
            #print(df_OB.head())
            #print(df_TR.head())

            #aggregate DATAFRAMES
            df_OB = df_OB.resample(str(sec)+'S',how={'spread':'mean','BidPrice_1':'mean','AskPrice_1':'mean','MidPrice_1':'mean','BidSize_1':'mean','AskSize_1':'mean'}).bfill()
            df_TR = df_TR.resample(str(sec)+'S',how={'side': 'sum', 'price': 'mean', 'sizeInBTC': 'sum', 'sizeInUSD': 'sum','nbOfTrades': 'sum'}).bfill()
            #print(df_OB.head())
            #print(df_OB.head())
            #print(df_TR.head())
            df_OB[['side']] = df_TR[['side']]
            df_OB[['price']] = df_TR[['price']]
            df_OB[['sizeInBTC']] = df_TR[['sizeInBTC']]
            df_OB[['sizeInUSD']] = df_TR[['sizeInUSD']]
            df_OB[['nbOfTrades']] = df_TR[['nbOfTrades']]
            df_OB = df_OB.dropna()
            #print(df_OB.head())
            # CALC OBI
            for x in range(0, len(df_OB)):
                x = df_OB.index[x]
                #print('x: ' + str(x))
                #df_OB.get_value
                #print('df_OB.get_value(x,"BidSize_1"):'+str(df_OB.get_value(x,'BidSize_1')))
                buff = (df_OB.get_value(x,'BidSize_1') - df_OB.get_value(x,'AskSize_1'))/(df_OB.get_value(x,'BidSize_1') + df_OB.get_value(x,'AskSize_1'))
                df_OB.set_value(x, 'OBI', buff)
            #CALC VOI INDICATOR
            NaNs = []
            for x in range(0, len(df_OB)):
                NaNs.append(np.NaN)
            df_OB['VOI'] = NaNs
            df_OB['RT'] = NaNs
            for x in range(1,len(df_OB)):
                prev_x = df_OB.index[x - 1]
                x = df_OB.index[x]

                prevBidPrice = df_OB.get_value(prev_x,'BidPrice_1')
                currBidPrice = df_OB.get_value(x,'BidPrice_1')
                prevAskPrice = df_OB.get_value(prev_x,'AskPrice_1')
                currAskPrice = df_OB.get_value(x,'AskPrice_1')

                prevBidVol = df_OB.get_value(prev_x,'BidSize_1')
                currBidVol = df_OB.get_value(x,'BidSize_1')
                prevAskVol = df_OB.get_value(prev_x,'AskSize_1')
                currAskVol = df_OB.get_value(x,'AskSize_1')

                if currBidPrice < prevBidPrice:
                    deltaVtB = 0
                elif currBidPrice == prevBidPrice:
                    deltaVtB = currBidVol - prevBidVol
                elif currBidPrice > prevBidPrice:
                    deltaVtB = currBidVol

                if currAskPrice < prevAskPrice:
                    deltaVtA = 0
                elif currAskPrice == prevAskPrice:
                    deltaVtA = currAskVol - prevAskVol
                elif currAskPrice > prevAskPrice:
                    deltaVtA = currAskVol

                VOI = deltaVtB - deltaVtA
                df_OB.set_value(x, 'VOI', VOI)

            #CALCULATE Rt (AVERAGE TRADE PRICE REGRESSION VALUE) LATER BUY AND SELL SIDE

            df_OB.set_value(df_OB.index[0], 'RT', df_OB.get_value(df_OB.index[0],'MidPrice_1'))
            for x in range(1, len(df_OB)):
                num = x
                prev_x = df_OB.index[x-1]
                x = df_OB.index[x]
                Mp = (df_OB.get_value(prev_x,'MidPrice_1') + df_OB.get_value(x,'MidPrice_1')) / 2


                nbTrades1 = df_OB.get_value(x,'nbOfTrades')
                nbTrades2 = df_OB.get_value(prev_x, 'nbOfTrades')
                #print ('nbTrades1: ' + str(nbTrades1))
                #print ('nbTrades2: ' + str(nbTrades2))

                if num == 1:
                    #print ('num: ' +str(num))
                    df_OB.set_value(x, 'RT', Mp)


                elif num > 1 and (nbTrades1 == nbTrades2):
                    #print ('num: ' + str(num))
                    buffVal = df_OB.get_value(prev_x, 'RT')
                    #print ('buffVal: ' + str(buffVal))
                    #print ('type(buffVal): ' + str(type(buffVal)))
                    df_OB.set_value(x, 'RT', buffVal)
                else:
                    #print ('else')
                    buff = ((df_OB.get_value(x,'sizeInBTC') - df_OB.get_value(prev_x,'sizeInBTC'))/(df_OB.get_value(x,'nbOfTrades')-df_OB.get_value(x-1,'nbOfTrades')))
                    import math
                    rt = (buff - Mp)
                    if rt>=0:
                        pass
                    else:
                        rt = -1*rt
                    #print ('rt: ' + str(rt))
                    df_OB.set_value(x,'RT',rt)
                    #df_OB.set_value(x,'RT') - Mp

            #df0.to_csv(dir + '/MICROSTRUCTURE_DATA.csv')

            '''
            DATA = []
            DATA_ACTUAL_PRICE = []
            dfTraining, dfTesting = self.splitdf(df0)
            if mode == 'TRAINING_V3':
                ##'bestMidPrice','chgMidPrice','spread'
                DATA_ACTUAL_PRICE = dfTraining[['bestBidPrice', 'bestAskPrice']]
                DATA = dfTraining[['spread', 'chgMidPrice', 'OBI_LVL_1', 'TSLBT', 'TSLST', 'BTSIZE', 'STSIZE']]
            elif mode == 'TESTING_V3':
                DATA_ACTUAL_PRICE = dfTesting[['bestBidPrice', 'bestAskPrice']]
                DATA = dfTesting[['spread', 'chgMidPrice', 'OBI_LVL_1', 'TSLBT', 'TSLST', 'BTSIZE', 'STSIZE']]
            else:
                print('INVALID MODE SPECIFIED')
                print('VALID MODES ARE: TRAINING/TESTING')
                exit()
            '''
            df_OB.to_csv(dir + '/MICROSTRUCTURE_DATA.csv')
            dfTraining, dfTesting = self.splitdf(df_OB)
            if mode == 'TRAINING':
                DATA_ACTUAL_PRICE = dfTraining[['BidPrice_1', 'AskPrice_1']]
                DATA = dfTraining[['OBI', 'RT', 'VOI']]
            elif mode == 'TESTING':
                DATA_ACTUAL_PRICE = dfTesting[['BidPrice_1', 'AskPrice_1']]
                DATA = dfTesting[['OBI', 'RT', 'VOI']]
            else:
                print('INVALID MODE SPECIFIED')
                print('VALID MODES ARE: TRAINING/TESTING')
                exit()
            return DATA, DATA_ACTUAL_PRICE
        else:
            dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
            df = pd.read_csv(dir + '/MICROSTRUCTURE_DATA.csv', index_col=['time'], parse_dates=['time'],
                              date_parser=dateparse)

            dfTraining, dfTesting = self.splitdf(df)
            if mode == 'TRAINING':
                DATA_ACTUAL_PRICE = dfTraining[['BidPrice_1', 'AskPrice_1']]
                DATA = dfTraining[['OBI', 'RT', 'VOI']]
            elif mode == 'TESTING':
                DATA_ACTUAL_PRICE = dfTesting[['BidPrice_1', 'AskPrice_1']]
                DATA = dfTesting[['OBI', 'RT', 'VOI']]
            else:
                print('INVALID MODE SPECIFIED')
                print('VALID MODES ARE: TRAINING/TESTING')
                exit()
            #print (DATA.tail(100))
            #exit()

            return DATA, DATA_ACTUAL_PRICE

    def loadOandaData(self, dir):
        # mode train/test dataset
        # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')
        # 2017-10-27T12:05:23.281Z
        # ORIG = pd.read_csv(dir + '/OrderBookData2.txt', index_col=['TIMESTAMP'], parse_dates=['TIMESTAMP'],date_parser=dateparse)
        # worker_threads = []
        lines = []
        with open(dir + '/AUD_USD_ORDERBOOK.txt', 'r') as f:
            alllines = f.readlines()
            # 'bestMidPrice','chgMidPrice','spread'
        for x in alllines:
            if x != '\n':
                lines.append(x)
        cols = ['time', 'midprice', 'OBBidPricelvl_1', 'OBBidLongCountlvl_1', 'OBBidShortCountlvl_1', 'OBAskPricelvl_1', 'OBAskLongCountlvl_1', 'OBAskShortCountlvl_1']
        npData = np.zeros((len(lines), len(cols)))
        # dfin = pd.DataFrame()
        dfin = pd.DataFrame(data=npData, columns=cols)  # dtype=np.float32

        lineNb = 0
        for line in lines:
            # line = f.readline()
            line = line.rstrip()
            print('lineNb: ' + str(lineNb))
            print('line: ' + str(line))
            if line == '':
                break

            json_data = ast.literal_eval(line)
            jsonLine = json.dumps(json_data)
            jsonLine = json.loads(jsonLine)
            #print('jsonLine:' + str(jsonLine))

            ts = datetime.strptime(jsonLine['orderBook']['time'], "%Y-%m-%dT%H:%M:%SZ") #2017-11-10T11:20:00Z'
            midPrice = float(jsonLine['orderBook']['price'])
            buckets = jsonLine['orderBook']['buckets']

            #loop through buckets to find closest bucket price to midPrice
            cols = ['midprice-bucketPrice','bucketPrice', 'shortCountPercent', 'longCountPercent']
            #print('len(buckets): ' + str(len(buckets)))
            npData = np.zeros((len(buckets), len(cols)))
            # dfin = pd.DataFrame()
            dfBuckets = pd.DataFrame(data=npData, columns=cols)  # dtype=np.float32

            i = 0
            for bucket in buckets:
                diff = midPrice - float(bucket['price'])
                dfBuckets.set_value(i,'midprice-bucketPrice', diff)
                dfBuckets.set_value(i, 'bucketPrice', (bucket['price']))
                dfBuckets.set_value(i, 'shortCountPercent', (bucket['shortCountPercent']))
                dfBuckets.set_value(i, 'longCountPercent', (bucket['longCountPercent']))

                i = i + 1
            #set to lvl0
            #neg - pos
            #ask - bid
            #print('dfBuckets: ' + str(dfBuckets))

            dfBuckets.sort_values('midprice-bucketPrice',ascending=True,axis=0,inplace=True,kind='qucksort')
            #print('dfBuckets: ' + str(dfBuckets))
            #lvl 0 closest to 0
            buff = 1000000000000

            lst = dfBuckets[['midprice-bucketPrice']].as_matrix()
            #print('lst: ' + str(lst))

            index = 0
            i =0
            for x in lst:
                x = x[0]
                #print('x: ' + str(x))

                if buff > x and x > 0:
                    buff = x
                    index = i
                i = i + 1
            bucket_price_lvl0 = buff
            #print (bucket_price_lvl0)
            #print ('bucket_price_lvl0: ' + str(bucket_price_lvl0))
            #print ('index: ' + str(index))

            nbLvls = 1
            #cols = ['time', 'midprice', 'OBBidPricelvl_1', 'OBBidLongCountlvl_1', 'OBBidShortCountlvl_1',
            #        'OBAskPricelvl_1', 'OBAskLongCountlvl_1', 'OBAskShortCountlvl_1']
            #dfin = pd.DataFrame(data=npData, columns=cols)  # dtype=np.float32

            #ask lvls
            #for i in range(bucket_price_lvl0,bucket_price_lvl0+nbLvls)
            #ts
            '''
            print('midPrice: ' + str(midPrice))
            print('dfBuckets["bucketPrice"][index+nbLvls]: ' + str(dfBuckets['bucketPrice'][index+nbLvls]))
            print('dfBuckets["longCountPercent"][index + nbLvls]: ' + str(dfBuckets['longCountPercent'][index + nbLvls]))
            print('dfBuckets["shortCountPercent"][index + nbLvls]: ' + str(dfBuckets['shortCountPercent'][index + nbLvls]))

            print('dfBuckets["bucketPrice"][index - nbLvls]: ' + str(dfBuckets['bucketPrice'][index - nbLvls]))
            print('dfBuckets["longCountPercent"][index - nbLvls]: ' + str(dfBuckets['longCountPercent'][index - nbLvls]))
            print('dfBuckets["shortCountPercent"][index - nbLvls]: ' + str(dfBuckets['shortCountPercent'][index - nbLvls]))
            '''
            dfin.set_value(index=lineNb, col='time', value=ts)
            dfin.set_value(index=lineNb, col='midprice', value=midPrice)

            dfin.set_value(index=lineNb,col='OBAskPricelvl_1',value=dfBuckets['bucketPrice'][index+nbLvls])
            dfin.set_value(index=lineNb, col='OBAskLongCountlvl_1', value=dfBuckets['longCountPercent'][index + nbLvls])
            dfin.set_value(index=lineNb, col='OBAskShortCountlvl_1', value=dfBuckets['shortCountPercent'][index + nbLvls])

            #for i in range(bucket_price_lvl0 - nbLvls, bucket_price_lvl0)
            dfin.set_value(index=lineNb, col='OBBidPricelvl_1', value=dfBuckets['bucketPrice'][index - nbLvls])
            dfin.set_value(index=lineNb, col='OBBidLongCountlvl_1', value=dfBuckets['longCountPercent'][index - nbLvls])
            dfin.set_value(index=lineNb, col='OBBidShortCountlvl_1', value=dfBuckets['shortCountPercent'][index - nbLvls])
            lineNb = lineNb + 1
        print('dfin: ' + str(dfin))

        exit()
            #get bucket x # of +/- 1 away from lvl 0



            # CALC OBI
            # OBI_LVL_1 = (float(jsonLine[0]['bids'][0][1]) - float(jsonLine[0]['asks'][0][1])) / (float(jsonLine[0]['bids'][0][1]) + float(jsonLine[0]['asks'][0][1]))
            #dfin.set_value(i, 'time', ts)
            #dfin.set_value(i, 'spread', float(jsonLine[0]['asks'][0][0]) - float(jsonLine[0]['bids'][0][0]))
            #dfin.set_value(i, 'BidPrice_1', float(jsonLine[0]['bids'][0][0]))
        '''
        df_OB = df_OB.set_index('time')
        df_TR = df_TR.set_index('time')
        # print(df_OB.head())
        # print(df_TR.head())

        # aggregate DATAFRAMES
        df_OB = df_OB.resample(str(sec) + 'S', how={'spread': 'mean', 'BidPrice_1': 'mean', 'AskPrice_1': 'mean',
                                                    'MidPrice_1': 'mean', 'BidSize_1': 'mean',
                                                    'AskSize_1': 'mean'}).bfill()
        df_TR = df_TR.resample(str(sec) + 'S',
                               how={'side': 'sum', 'price': 'mean', 'sizeInBTC': 'sum', 'sizeInUSD': 'sum',
                                    'nbOfTrades': 'sum'}).bfill()
        # print(df_OB.head())
        # print(df_OB.head())
        # print(df_TR.head())
        # CALC OBI
        for x in range(0, len(df_OB)):
            x = df_OB.index[x]
            # print('x: ' + str(x))
            # df_OB.get_value
            # print('df_OB.get_value(x,"BidSize_1"):'+str(df_OB.get_value(x,'BidSize_1')))
            buff = (df_OB.get_value(x, 'BidSize_1') - df_OB.get_value(x, 'AskSize_1')) / (
            df_OB.get_value(x, 'BidSize_1') + df_OB.get_value(x, 'AskSize_1'))
            df_OB.set_value(x, 'OBI', buff)


        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        df = pd.read_csv(dir + '/MICROSTRUCTURE_DATA.csv', index_col=['time'], parse_dates=['time'],
                         date_parser=dateparse)

        dfTraining, dfTesting = self.splitdf(df)

        #return DATA, DATA_ACTUAL_PRICE
        '''
        return

    def convertRawOrderBookToDF(self,dir):
        # mode train/test dataset
        # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')
        # 2017-10-27T12:05:23.281Z
        # ORIG = pd.read_csv(dir + '/OrderBookData2.txt', index_col=['TIMESTAMP'], parse_dates=['TIMESTAMP'],date_parser=dateparse)

            with open(dir + '/OrderBook.txt', 'r') as f:
                lines = f.readlines()
                # 'bestMidPrice','chgMidPrice','spread'
            cols = ['time','spread','BidPrice_1', 'AskPrice_1', 'MidPrice_1','BidSize_1', 'AskSize_1']
            npData = np.zeros((len(lines), len(cols)))
            # dfin = pd.DataFrame()
            dfin = pd.DataFrame(data=npData, columns=cols)  # dtype=np.float32

            i = 0
            for line in lines:
                # line = f.readline()
                line = line.rstrip()
                if line == '':
                    break

                json_data = ast.literal_eval(line)
                jsonLine = json.dumps(json_data)
                jsonLine = json.loads(jsonLine)

                ts = datetime.strptime(jsonLine[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")

                # CALC OBI
                #OBI_LVL_1 = (float(jsonLine[0]['bids'][0][1]) - float(jsonLine[0]['asks'][0][1])) / (float(jsonLine[0]['bids'][0][1]) + float(jsonLine[0]['asks'][0][1]))
                dfin.set_value(i, 'time', ts)
                dfin.set_value(i, 'spread', float(jsonLine[0]['asks'][0][0]) - float(jsonLine[0]['bids'][0][0]))
                dfin.set_value(i, 'BidPrice_1', float(jsonLine[0]['bids'][0][0]))
                dfin.set_value(i, 'AskPrice_1', float(jsonLine[0]['asks'][0][0]))
                dfin.set_value(i, 'MidPrice_1', (float(jsonLine[0]['asks'][0][0]) + float(jsonLine[0]['bids'][0][0]))/2)
                dfin.set_value(i, 'BidSize_1', float(jsonLine[0]['bids'][0][1]))
                dfin.set_value(i, 'AskSize_1', float(jsonLine[0]['asks'][0][1]))
                i = i + 1
            return dfin
    def convertRawLatestTradesToDF(self,dir):
        # mode train/test dataset
        # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')
        # 2017-10-27T12:05:23.281Z
        # ORIG = pd.read_csv(dir + '/OrderBookData2.txt', index_col=['TIMESTAMP'], parse_dates=['TIMESTAMP'],date_parser=dateparse)

            with open(dir + '/LatestTrades.txt', 'r') as f:
                lines = f.readlines()
            cols = ['time','side','price', 'sizeInBTC', 'sizeInUSD','nbOfTrades']
            npData = np.zeros((len(lines), len(cols)))
            # dfin = pd.DataFrame()
            dfin = pd.DataFrame(data=npData, columns=cols)  # dtype=np.float32

            i = 0
            for line in lines:
                # line = f.readline()
                line = line.rstrip()
                if line == '':
                    break

                json_data = ast.literal_eval(line)
                jsonLine = json.dumps(json_data)
                jsonLine = json.loads(jsonLine)

                ts = datetime.strptime(jsonLine['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
                price = float(jsonLine['price'])
                side = jsonLine['side']
                sizeInBTC = float(jsonLine['homeNotional'])
                sizeInUSD = float(jsonLine['size'])
                dfin.set_value(i, 'time', ts)
                if side =='Sell':
                    dfin.set_value(i, 'side', -1.00)
                else:
                    dfin.set_value(i, 'side', 1.00)
                dfin.set_value(i, 'price', price)
                dfin.set_value(i, 'sizeInBTC', sizeInBTC)
                dfin.set_value(i, 'sizeInUSD', sizeInUSD)
                dfin.set_value(i, 'nbOfTrades', 1.00)

                i = i + 1
            return dfin
    def addTimeSinceLastTrade_to_DF(self,dir,df,sideMode):
        ob = 0
        lastBTts = 0
        #nbLines = 0
        isReadNextline = True
        isEOF = False
        #with open('./bitmex_data/LatestTrades.txt', 'r') as f:
        #    totLines = f.readlines()
        #nbLines = len(totLines)
        #totLines = None
        #print('REACH HERE')
        with open(dir+'/LatestTrades.txt', 'r') as f:
            currLinenb = 0
            while isEOF == False:
                if isReadNextline == True:
                    isEOF, side, size, BTts = self.getNextLastestTradesLine(f, sideMode)
                    #print(str(nbLines) + str(' <-> ') + str(currLinenb))
                    currLinenb = currLinenb + 1
                    isReadNextline = False
                    if isEOF == True:
                        break

                OBts = df.get_value(ob, 'time') #DF DATE FORMAT

                '''
                print('type(OBts): ' + str(type(OBts)))
                print('type(BTts): ' + str(type(BTts)))

                print('OBts: ' + str(OBts))
                print('BTts: ' + str(BTts))
                '''
                if OBts < BTts and lastBTts == 0:
                    #print('REACH HERE_1')
                    # remove row from DF
                    ob = ob + 1
                    df = df[1:]

                elif OBts < BTts:
                    #print('REACH HERE_3')
                    # add tslbt to current Ob DF row
                    tslbt = (OBts - lastBTts).microseconds  # convert to ms float?
                    if sideMode == 'Buy':
                        df.set_value(ob, 'TSLBT', tslbt)
                        df.set_value(ob, 'BTSIZE', size)  # TimeSinceLastBuytrade
                    else:
                        df.set_value(ob, 'TSLST', tslbt)  # TimeSinceLastBuytrade
                        df.set_value(ob, 'STSIZE', size)  # TimeSinceLastBuytrade

                    # df.set_value('ROW', 'COL', VAL)            #TimeSinceLastBuytrade
                    ob = ob + 1

                elif OBts >= BTts:
                    #print('REACH HERE_2')
                    lastBTts = BTts
                    tslbt = (OBts - lastBTts).microseconds
                    #print('tslbt: ' + str(tslbt))
                    #exit()
                    # BTsize = lastBTsize
                    if sideMode == 'Buy':
                        df.set_value(ob, 'TSLBT', tslbt)  # TimeSinceLastBuytrade
                        df.set_value(ob, 'BTSIZE', size)  # TimeSinceLastBuytrade
                    else:
                        df.set_value(ob, 'TSLST', tslbt)  # TimeSinceLastBuytrade
                        df.set_value(ob, 'STSIZE', size)  # TimeSinceLastBuytrade
                    ob = ob + 1
                    isReadNextline = True

                    # Save Dataframe to file

        return df

    def getNextLastestTradesLine(self,f,choosenSide):

        line = f.readline()
        line = line.rstrip()
        if line == '':
            return True,'','',''

        json_data = ast.literal_eval(line)
        jsonLine = json.dumps(json_data)
        jsonLine = json.loads(jsonLine)

        side = jsonLine['side']
        size = int(jsonLine['size'])
        ts = datetime.strptime(jsonLine['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
        #print(str(ts))
        if  choosenSide != side:
            #read the next line
            self.getNextLastestTradesLine(f,choosenSide)

        return False, side, size, ts

    def priceRange(self, NORM_PV, bins):
        # PRICE RANGE

        bidPrices = NORM_PV['bidPrice_MEAN'].as_matrix()
        askPrices = NORM_PV['askPrice_MEAN'].as_matrix()

        midPrice = (askPrices + bidPrices) / 2.00
        PriceRange = np.zeros((len(midPrice), len(bins)), dtype=float)

        for t in range(0, len(midPrice)):
            for x in range(0, len(bins)):
                [lowEdge, highEdge] = bins[x]
                if lowEdge <= midPrice[t] and highEdge >= midPrice[t]:
                    PriceRange[t][x] = 1.00

        print ('PriceRange: ' + str(PriceRange))
        return PriceRange,midPrice

    def rolStdDevRange(self, NORM_PV, bins):
        # PRICE RANGE
        npData = (NORM_PV['askPrice_MEAN'].as_matrix()) + (NORM_PV['bidPrice_MEAN'].as_matrix())
        midPrice = pd.DataFrame(data=npData, columns=['midPrice_MEAN'], index=NORM_PV.index, dtype=np.float32)
        midPrice = NORM(MODE='STDDEV').IN(df=midPrice,windowSize=1)
        npMidPrice = midPrice.as_matrix()
        PriceRange = np.zeros((len(npMidPrice), len(bins)), dtype=float)

        for t in range(0, len(npMidPrice)):
            for x in range(0, len(bins)):
                [lowEdge, highEdge] = bins[x]
                if lowEdge <= npMidPrice[t] and highEdge >= npMidPrice[t]:
                    PriceRange[t][x] = 1.00

        print ('rolStdDevRange: ' + str(PriceRange))
        return PriceRange,midPrice

    def totalVolRange(self,NORM_PV,bins):
        # PRICE RANGE

        bidSize = NORM_PV['bidSize_MEAN'].as_matrix()
        askSize = NORM_PV['askSize_MEAN'].as_matrix()

        totVolSize = (bidSize + askSize)/2.00
        totVolRange = np.zeros((len(totVolSize), len(bins)), dtype=float)

        for t in range(0, len(totVolSize)):
            for x in range(0, len(bins)):
                [lowEdge, highEdge] = bins[x]
                if lowEdge <= totVolSize[t] and highEdge >= totVolSize[t]:
                    totVolRange[t][x] = 1.00

        print ('totVolRange: ' + str(totVolRange))
        return totVolRange,totVolSize

    def BidAskVolRange(self,NORM_PV,bins):
        npData = (NORM_PV['bidSize_MEAN'].as_matrix()) - (NORM_PV['askSize_MEAN'].as_matrix())
        totVolSize = pd.DataFrame(data=npData, columns=['bidaskvoldiff_MEAN'], index=NORM_PV.index, dtype=np.float32)

        totVolSize = NORM(MODE='STD').IN(df=totVolSize,windowSize=1)
        npTotVolSize = totVolSize.as_matrix()
        BidAskVolRange = np.zeros((len(npTotVolSize), len(bins)), dtype=float)

        for t in range(0, len(npTotVolSize)):
            for x in range(0, len(bins)):
                [lowEdge, highEdge] = bins[x]
                if lowEdge <= npTotVolSize[t] and highEdge >= npTotVolSize[t]:
                    BidAskVolRange[t][x] = 1.00

        print ('BidAskVolRange: ' + str(BidAskVolRange))
        return BidAskVolRange,totVolSize

    def bidAskSpreadrange(self, NORM_PV, bins):
        # PRICE RANGE
        npData = (NORM_PV['askPrice_MEAN'].as_matrix()) - (NORM_PV['bidPrice_MEAN'].as_matrix())
        midPrice = pd.DataFrame(data=npData, columns=['midPrice_MEAN'], index=NORM_PV.index, dtype=np.float32)

        midPrice = NORM(MODE='STD').IN(df=midPrice,windowSize=1)
        npMidPrice = midPrice.as_matrix()

        PriceRange = np.zeros((len(npMidPrice), len(bins)), dtype=float)

        for t in range(0, len(npMidPrice)):
            for x in range(0, len(bins)):
                [lowEdge, highEdge] = bins[x]
                if lowEdge <= npMidPrice[t] and highEdge >= npMidPrice[t]:
                    PriceRange[t][x] = 1.00

        print ('bidAskSpreadrange: ' + str(PriceRange))
        return PriceRange,midPrice


    def loadBitmexDFFilesV2(self, dir, tDelta, resolution):
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

        #ORIG_P_NEW
        #NORM_PV_NEW
        ORIG_P_1 = pd.read_csv(dir + '/ORIG_P.csv', index_col=['time'], parse_dates=['time'], date_parser=dateparse)
        NORM_PV_1 = pd.read_csv(dir + '/NORM_PV.csv', index_col=['time'], parse_dates=['time'], date_parser=dateparse)

        NORM_PV_1 = NORM_PV_1[[
            'bidPrice_MEAN',
            'askPrice_MEAN',
            'bidSize_MEAN',
            'askSize_MEAN',
        ]]

        print (str(len(ORIG_P_1)))
        ORIG_P = self.oversample_df_OHLCv2(ORIG_P_1, tDelta, 1)  # 10 mins
        NORM_PV = self.oversample_df_OHLCv2(NORM_PV_1, tDelta, 1)  # 10 mins
        ORIG_P = self.ZeroOutNaNs(ORIG_P[0])
        NORM_PV = self.ZeroOutNaNs(NORM_PV[0])
        print (str(len(ORIG_P)))
        low = -1.00
        high = 1.00
        totalRange = 2.00

        binStep = totalRange/float(resolution)
        bins = []
        l = -1.00
        h = -1.00+binStep
        for x in range(0,resolution):
            bins.append([l+(binStep*x),h+(binStep*x)])
        #print ('bins: ' + str(bins))

        # PRICE RANGE
        PriceRanges,npData1 = self.priceRange(NORM_PV, bins)

        # TOTAL VOLUME
        TotalVolumeRanges,npData2 = self.totalVolRange(NORM_PV, bins)

        # BID ASK VOLUME
        BidAskVolumeRange,npData3 = self.BidAskVolRange(NORM_PV, bins)

        # ROL STD DEV RANGE
        rolStdDevRange,npData4 = self.rolStdDevRange(NORM_PV, bins)

        #BID ASK SPREAD RANGE
        bidAskSpreadrange,npData5 = self.bidAskSpreadrange(NORM_PV, bins)

        print(PriceRanges.shape)
        print(TotalVolumeRanges.shape)
        print(BidAskVolumeRange.shape)
        print(rolStdDevRange.shape)
        print(bidAskSpreadrange.shape)

        npData = np.append(PriceRanges, TotalVolumeRanges, axis=1)
        npData = np.append(npData, BidAskVolumeRange, axis=1)
        npData = np.append(npData, rolStdDevRange, axis=1)
        npData = np.append(npData, bidAskSpreadrange, axis=1)
        length,width = npData.shape
        npData = npData.reshape((length,5,resolution))
        #print(npData)
        #for x in range(0,length):
        #    print(npData[x])

        #exit()
        return ORIG_P, npData

    def loadForexDFFiles(self, dir, tDelta, nbSamp):
        #AUD / NZD, 20170901 00:00:00.735, 1.10624, 1.10638

        dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d %H:%M:%S.%f')
        #pair,time,bid,ask
        ORIG_P_1 = pd.read_csv(dir + '/AUDNZD-2017-09.csv', index_col=['time'], parse_dates=['time'], date_parser=dateparse)
        #NORM_PV_1 = pd.read_csv(dir + '/NORM_PV_NEW.csv', index_col=['time'], parse_dates=['time'], date_parser=dateparse)

        ORIG_P_1 = ORIG_P_1[['bid','ask']]
        print('1: ' +str(ORIG_P_1.head()))
        if nbSamp>0:
            ORIG_P_OS_LST_1  = self.oversample_df_OHLCv2(ORIG_P_1, tDelta, nbSamp)  # 10 mins
            #NORM_PV_OS_LST_1 = self.oversample_df_OHLCv2(NORM_PV_1, tDelta, nbSamp)  # 10 mins
            for x in range(0,len(ORIG_P_OS_LST_1)):
                ORIG_P_OS_LST_1[x] = self.ZeroOutNaNs(ORIG_P_OS_LST_1[x])
                #NORM_PV_OS_LST_1[x] = self.ZeroOutNaNs(NORM_PV_OS_LST_1[x])
                print('2: ' + str(ORIG_P_OS_LST_1[x]))

            return ORIG_P_OS_LST_1
        else:
            ORIG_P_1 = self.ZeroOutNaNs(ORIG_P_1)
            return ORIG_P_1
        #print(ORIG_P_OS_LST_1)
        #print(NORM_PV_OS_LST_1)


    #bitmex
    #### - HELPER FUNC - #####
    def resample_df_OHLC(self,df):
        # second longest time suck
        start = datetime.datetime.utcnow()
        df_resampled = self.dataframeAggrv2(df, self.TimeDeltaPerDataPoint)
        end = datetime.datetime.utcnow()
        resampleTime = (end - start).total_seconds()
        print('resampleTime: ' + str(resampleTime))
        return df_resampled
    def oversample_df_OHLCv2(self,df,tDelta,nbSamp):
        # longest time suck
        start = datetime.utcnow()

        if int(tDelta) % int(nbSamp) == 0 and (tDelta>=nbSamp):
            overSampleRate = tDelta / nbSamp
        else:
            print('ERROR: invalid oversampling value')
            return -1

        df_oversampled = self.dataframeAggrv2(df, overSampleRate)

        oversampled_DFs = []
        for startIndex in range(0, int(nbSamp)):
            oversampled_DFs.append(df_oversampled.iloc[startIndex::int(nbSamp),
                                   :])  # select ever 5th row starting from the first row (0,5,10) and start from 2nd row (1,6,11),3rd row (2,7,12)...(4,9,14)

        end = datetime.utcnow()
        oversampleTime = (end - start).total_seconds()
        print('oversampleTime: ' + str(oversampleTime))
        return oversampled_DFs
    def dataframeAggrv2(self,df, tDelta):
        # BIGGEST TIME SUCK
        '''
        These operations are supported by pandas.eval():

        Arithmetic operations except for the left shift (<<) and right shift (>>) operators, e.g., df + 2 * pi / s ** 4 % 42 - the_golden_ratio
        Comparison operations, including chained comparisons, e.g., 2 < df < df2
        Boolean operations, e.g., df < df2 and df3 < df4 or not df_bool
        list and tuple literals, e.g., [1, 2] or (1, 2)
        Attribute access, e.g., df.a
        Subscript expressions, e.g., df[0]
        Simple variable evaluation, e.g., pd.eval('df') (this is not very useful)
        Math functions, sin, cos, exp, log, expm1, log1p, sqrt, sinh, cosh, tanh, arcsin, arccos, arctan, arccosh, arcsinh, arctanh, abs and arctan2.
        '''
        # [start:end:increment]
        nbRow = df.shape[0]  # gives number of row count
        # nbCol = df.shape[1]  # gives number of col coun
        # select ever 5th row starting from the first row (0,5,10)
        # and start from 2nd row (1,6,11),3rd row (2,7,12)...(4,9,14)
        tCloseIndexes = df.index  # [0::int(tDelta), :]  # every tDelta Row .to_pydatetime()
        # print('len(tCloseIndexes): ' + str(len(tCloseIndexes)))
        # print('tDelta: ' + str(tDelta))
        tCloseIndexAggr = [tCloseIndexes[x] for x in range(0, nbRow, int(tDelta))]
        # print('tCloseIndexes: ' + str(tCloseIndexes))
        # print('tCloseIndexAggr: ' + str(tCloseIndexAggr))
        # print('len(tCloseIndexes)  : ' + str(len(tCloseIndexes)))
        # print('df nbRow: ' + str(nbRow))
        # print('len(df): ' + str(len(df)))
        firstDate = str(df.head(1).index.tolist()[0].to_pydatetime())

        rng = pd.date_range(firstDate, periods=nbRow, freq='1T')  # 1min
        df = df.set_index(rng)

        # tDelta =  #1min = Row
        # AIM: CHANGE 1MIN ARRAY INTO TDELTA ARRAY
        # organise columns for resample rule set


        # print('len(df): ' + str(len(df)))
        df = df.resample(str(int(tDelta)) + 'T', closed='left', label='left').mean()    #asfreq()
        # df = df.resample(str(int(tDelta)) + 'T').agg(how=dict_ohlc, closed='left', label='left')
        npArray = df.as_matrix()
        npArray = np.resize(npArray, (len(tCloseIndexAggr), len(df.columns)))
        # print ('npArray.shape: ' + str(npArray.shape))
        df = pd.DataFrame(npArray, columns=df.columns, dtype=float, index=tCloseIndexAggr)  #
        # df #(row,col)
        # print('len(df): ' + str(len(df)))
        # print('len(tCloseIndexAggr): ' + str(len(tCloseIndexAggr)))
        # df['TIME_CLOSE'] = tCloseIndexAggr
        # df = df.set_index('TIME_CLOSE')
        # print('df: ' + str(df))
        # df = df.set_index(tCloseIndexAggr, inplace=True)   #.to_pydatetime()

        return df
    def ZeroOutNaNs(self,df):
        npData = df.as_matrix()
        rows,cols = npData.shape

        for row in range(0,rows):
            for col in range(0,cols):
                if str(npData[row][col]) == 'nan':
                    #print('npData[' + str(row-1) + '][' + str(col) + ']: ' + str(npData[row-1][col]))
                    print('NAN FOUND npData['+ str(row)+']['+ str(col)+']: ' + str(npData[row][col]))
                    print('ERROR NAN FOUND')
                    npData[row][col] = 0.0
                    #exit()
                    #return False
        df_out = pd.DataFrame(data=npData, columns=df.columns, index=df.index, dtype=np.float32)

        return df_out

    def isTSLTriggered(self, side, firstRun, currPrice):

            if self.TSLOffset == 0:
                print('Bad offset value given.')
                exit()
            elif self.TSLOffset < 0:
                self.TSLOffset = -1.00 * self.TSLOffset

            if side == 'Long':
                # SET rollingSL
                if firstRun == True:
                    self.rollingSL = currPrice - self.TSLOffset
                    return False, currPrice

                # rollingRL triggered
                elif currPrice <= self.rollingSL:
                    return True, currPrice

                # Tighten TSLprice/rollingSL
                elif (currPrice - self.rollingSL) > self.TSLOffset:
                    self.rollingSL = currPrice - self.TSLOffset
                    return False, currPrice
                else:
                    return False, currPrice

            elif side == 'Short':
                # SET rollingSL
                if firstRun == True:
                    self.rollingSL = currPrice + self.TSLOffset
                    return False, currPrice

                # rollingRL triggered
                elif currPrice >= self.rollingSL:
                    return True, currPrice

                # Tighten TSLprice/rollingSL
                elif (self.rollingSL - currPrice) > self.TSLOffset:
                    self.rollingSL = currPrice + self.TSLOffset
                    return False, currPrice
                else:
                    return False, currPrice
            else:
                print ('Bad trade side value given')
                exit()

                #test = Bitmex()

class BitmexTest:
    def __init__(self,VolThresh,dir,lag, mode, tDelay, loadPrevDF,TSLOffset,hist_Vals):
        self.VolThresh = VolThresh
        self.hist_Vals  = hist_Vals
        self.rollingSL = 0
        self.runningTot = 0
        self.TSLOffset = TSLOffset
        self.nbToIgnore = 0
        self.VolThreshIterationVals = []
        self.DATA_VOL = pd.DataFrame()
        self.DATA_VOL_NORM= pd.DataFrame()
        self.DATA_ACTUAL_PRICE_VOL = pd.DataFrame()
        self.DATA_RAW = pd.DataFrame()
        self.DATA_RAW_NORM = pd.DataFrame()
        self.DATA_ACTUAL_PRICE_RAW = pd.DataFrame()
        self.percentBias = 0.3
        self.NORMOBJ = NORM(MODE='STD')
        self.setupNormalise(dir, lag, mode, tDelay, loadPrevDF)
        with open(dir + '/OrderBook.txt', 'r') as f:
            self.lines = f.readlines()

        line = self.lines[0].rstrip()

        json_data = ast.literal_eval(line)
        jsonLine = json.dumps(json_data)
        jsonLine = json.loads(jsonLine)
        self.curr_ts = datetime.strptime(jsonLine[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
        if mode == 'training':
            self.i = 0
            self.end = len(self.lines) / 2
        else:
            self.i = len(self.lines) / 2
            self.end = len(self.lines)
        return

    def splitdf(self,df):

        npData=df.as_matrix()
        nbRows=len(npData)
        nbSplitRows=0
        if (nbRows % 2) == 0:
            nbSplitRows = nbRows/2
        else:
            nbSplitRows = (nbRows-1) / 2

        df_1 = pd.DataFrame(data=npData[0:nbSplitRows], columns=df.columns, index=df.index[0:nbSplitRows], dtype=np.float32)
        df_2 = pd.DataFrame(data=npData[nbSplitRows:], columns=df.columns, index=df.index[nbSplitRows:],dtype=np.float32)

        return df_1,df_2
    def reset(self):
        self.i = 0
    def iterateOBdataV2(self):
        #print ('iterateOBdataV2')
        for a in range(0,len(self.VolThreshIterationVals) - self.nbToIgnore):
            #print('len(self.VolThreshIterationVals): ' + str(len(self.VolThreshIterationVals)))
            if len(self.VolThreshIterationVals)- self.nbToIgnore>=a+self.nbToIgnore:
                #print('len(self.VolThreshIterationVals): ' + str(len(self.VolThreshIterationVals)))
                x = self.VolThreshIterationVals[a+self.nbToIgnore]
            else:
                return []
            if self.i <= x:

                ret = []
                for h in range(-self.hist_Vals,0,1):
                    #print(self.DATA_ACTUAL_PRICE_VOL.index)
                    #print(str(x) + ' out of ' + str(self.VolThreshIterationVals[-1]))

                    #print(str(self.runningTot) +' out of ' + str(len(self.VolThreshIterationVals)))
                    #self.runningTot = self.runningTot + 1
                    self.bid = self.DATA_ACTUAL_PRICE_RAW.get_value(index=x+h,col='bidP')#[self.i]
                    self.ask = self.DATA_ACTUAL_PRICE_RAW.get_value(index=x+h,col='askP')

                    #print('df: ' + str(df))
                    #print('self.bid: ' + str(self.bid))
                    #print('self.ask: ' + str(self.ask))
                    #cols = ['spread', 'chgMidP', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5', 'bidS_1','askS_1']

                    spread = self.DATA_RAW_NORM.get_value(index=x+h,col='spread')
                    chgMidP = self.DATA_RAW_NORM.get_value(index=x+h,col='chgMidP')
                    OBI_LVL_1 = self.DATA_RAW_NORM.get_value(index=x+h,col='OBI_LVL_1')
                    OBI_LVL_2 = self.DATA_RAW_NORM.get_value(index=x+h,col='OBI_LVL_2')
                    OBI_LVL_3 = self.DATA_RAW_NORM.get_value(index=x+h,col='OBI_LVL_3')
                    OBI_LVL_4 = self.DATA_RAW_NORM.get_value(index=x+h,col='OBI_LVL_4')
                    OBI_LVL_5 = self.DATA_RAW_NORM.get_value(index=x+h,col='OBI_LVL_5')
                    bidS_1 = self.DATA_RAW_NORM.get_value(index=x+h,col='bidS_1')
                    askS_1 = self.DATA_RAW_NORM.get_value(index=x+h,col='askS_1')

                    DATA = [spread,chgMidP,OBI_LVL_1,OBI_LVL_2,OBI_LVL_3,OBI_LVL_4,OBI_LVL_5,bidS_1,askS_1]
                    ret.append(DATA)
                DATA = np.asarray(ret)

                self.i = x+1
                return DATA
        return []
    def iterateOBdataIgnoringVolThreshV2(self):
        #print ('iterateOBdataIgnoringVolThreshV2')
        for _ in range(self.i, len(self.DATA_ACTUAL_PRICE_RAW)):
            self.bid = self.DATA_ACTUAL_PRICE_RAW.get_value(index=self.i, col='bidP')  # [self.i]
            self.ask = self.DATA_ACTUAL_PRICE_RAW.get_value(index=self.i, col='askP')

            spread = self.DATA_RAW_NORM.get_value(index=self.i, col='spread')
            chgMidP = self.DATA_RAW_NORM.get_value(index=self.i, col='chgMidP')
            OBI_LVL_1 = self.DATA_RAW_NORM.get_value(index=self.i, col='OBI_LVL_1')
            OBI_LVL_2 = self.DATA_RAW_NORM.get_value(index=self.i, col='OBI_LVL_2')
            OBI_LVL_3 = self.DATA_RAW_NORM.get_value(index=self.i, col='OBI_LVL_3')
            OBI_LVL_4 = self.DATA_RAW_NORM.get_value(index=self.i, col='OBI_LVL_4')
            OBI_LVL_5 = self.DATA_RAW_NORM.get_value(index=self.i, col='OBI_LVL_5')
            bidS_1 = self.DATA_RAW_NORM.get_value(index=self.i, col='bidS_1')
            askS_1 = self.DATA_RAW_NORM.get_value(index=self.i, col='askS_1')

            DATA = [spread, chgMidP, OBI_LVL_1, OBI_LVL_2, OBI_LVL_3, OBI_LVL_4, OBI_LVL_5, bidS_1, askS_1]
            DATA = np.asarray(DATA)

            self.i = self.i + 1
            return DATA
        return []

    def normalise(self, df):
        #print ('normalise')
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
            # NORM
            df_Out = self.NORMOBJ.IN(df_Out, isLog=False)
            return True, df_Out
        else:
            return False, ''
    def normaliseWithoutVolThresh(self, df):
        #print ('normaliseWithoutVolThresh')

        df_Out = self.NORMOBJ.IN(df, isLog=False)
        return df_Out
    def getLatestBidAsk(self):

        return float(self.bid), float(self.ask)

    def getLatest1SecData(self):

        DATA = self.iterateOBdataV2()
        return DATA

    def iterateDataIgnoringVolThresh(self):

        DATA = self.iterateOBdataIgnoringVolThreshV2()
        return DATA


    def setupNormalise(self,dir, lag, mode, tDelay,loadPrevDF):
        BMX = Bitmex()

        DATA, DATA_ACTUAL_PRICE = BMX.collectDatav3(dir, lag, mode, tDelay, loadPrevDF)
        self.DATA_RAW = DATA
        self.DATA_ACTUAL_PRICE_RAW = DATA_ACTUAL_PRICE
        '''
        DATA_ACTUAL_PRICE['pcentPriceChgLong'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['pcentPriceChgShort'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['entryPriceLong'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['entryPriceShort'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['exitPriceLong'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['exitPriceShort'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['entryIndex'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['exitIndexLong'] = DATA['chgMidP']
        DATA_ACTUAL_PRICE['exitIndexShort'] = DATA['chgMidP']

        '''
        #DATA_ACTUAL_PRICE_before_reduction = DATA_ACTUAL_PRICE

        newNpData = []
        newNpDataIndexes = []
        newNpDataActual = []
        newNpDataIndexesActual = []
        npData = DATA.as_matrix()
        npDataActual = DATA_ACTUAL_PRICE.as_matrix()
        indexVals = DATA.index.values
        indexValsActual = DATA_ACTUAL_PRICE.index.values
        for x in range(0, len(npData)):
            if npData[x][7] > self.VolThresh or npData[x][8] > self.VolThresh:
                newNpData.append(npData[x])
                newNpDataActual.append(npDataActual[x])
                newNpDataIndexes.append(indexVals[x])
                newNpDataIndexesActual.append(indexValsActual[x])
                self.VolThreshIterationVals.append(x)

        npData = np.asarray(newNpData, dtype=float)
        df_Out = pd.DataFrame(data=npData, columns=DATA.columns, index=newNpDataIndexes, dtype=np.float32)
        df_OutActual = pd.DataFrame(data=newNpDataActual, columns=DATA_ACTUAL_PRICE.columns,
                                    index=newNpDataIndexesActual, dtype=np.float32)
        #df_Out
        DATA = df_Out
        DATA_ACTUAL_PRICE = df_OutActual
        self.DATA_VOL = DATA
        self.DATA_ACTUAL_PRICE_VOL = DATA_ACTUAL_PRICE

        print('***SETUP NORMALISATION***')
        print('len(DATA_ACTUAL_PRICE): ' + str(len(DATA_ACTUAL_PRICE)))
        print('len(DATA): ' + str(len(DATA)))
        print('self.VolThreshIterationVals: ' + str(self.VolThreshIterationVals))
        DATA = DATA.dropna()
        self.DATA_RAW_NORM = self.NORMOBJ.IN(self.DATA_RAW, isLog=False)
        self.DATA_VOL_NORM = self.NORMOBJ.IN(DATA, isLog=False)

    def setTSL(self,TSL):
        self.TSLOffset = TSL
    def setnbToIgnore(self,nbToIgnore):
        self.nbToIgnore = nbToIgnore
    def goLong(self):
        #print ('goLong')
        bid, ask = self.getLatestBidAsk()
        #ask
        side = 'Long'
        firstRun = True
        isTSLTrigged = False
        currPrice = ask
        entryPrice = currPrice
        entryIndex = self.i
        prevBid = 0
        while isTSLTrigged == False:
            isTSLTrigged, slPrice = self.isTSLTriggered(side, firstRun, currPrice)

            if firstRun == False:
                while prevBid == bid:
                    self.iterateDataIgnoringVolThresh()  # iterate data
                    bid, ask = self.getLatestBidAsk()  # get next price from top of OB

            else:
                self.iterateDataIgnoringVolThresh() #iterate data
                bid, ask = self.getLatestBidAsk() #get next price from top of OB

            firstRun = False
            currPrice = bid
            prevBid = bid

        exitPrice = self.rollingSL
        exitIndex = self.i
        #price percentage change
        pcentPriceChgLong = (((exitPrice - entryPrice) / entryPrice) * 100) - self.percentBias
        #pcentPriceChgShort = ((minExitPriceShort - entryPriceShort) / entryPriceShort) * 100

        #reward metric
        actions=[side,entryIndex,exitIndex,entryPrice,exitPrice,pcentPriceChgLong]
        return pcentPriceChgLong,actions

    def goShort(self):
        #print ('goShort')
        bid, ask = self.getLatestBidAsk()
        #ask
        side = 'Short'
        firstRun = True
        isTSLTrigged = False
        currPrice = bid
        entryPrice = currPrice
        entryIndex = self.i
        prevAsk = 0
        while isTSLTrigged == False:
            isTSLTrigged, slPrice = self.isTSLTriggered(side, firstRun, currPrice)
            if firstRun == False:
                while prevAsk == ask:
                    self.iterateDataIgnoringVolThresh()  # iterate data
                    bid, ask = self.getLatestBidAsk()  # get next price from top of OB
                    #prevAsk = ask
            else:
                self.iterateDataIgnoringVolThresh() #iterate data
                bid, ask = self.getLatestBidAsk() #get next price from top of OB

            currPrice = ask
            prevAsk = ask

            firstRun = False

        exitPrice = self.rollingSL
        exitIndex = self.i
        #price percentage change
        pcentPriceChgShort = (-1.00*(((exitPrice - entryPrice) / entryPrice) * 100)) - self.percentBias
        #self.percentBias = 0.3
        #pcentPriceChgShort = ((minExitPriceShort - entryPriceShort) / entryPriceShort) * 100

        #reward metric
        actions = [side,entryIndex, exitIndex, entryPrice, exitPrice,pcentPriceChgShort]

        return pcentPriceChgShort,actions

    def isTSLTriggered(self, side, firstRun, currPrice):

            #print ('isTSLTriggered')
            if self.TSLOffset == 0:
                print('Bad offset value given.')
                exit()
            elif self.TSLOffset < 0:
                self.TSLOffset = -1.00 * self.TSLOffset

            if side == 'Long':
                # SET rollingSL
                if firstRun == True:
                    self.rollingSL = currPrice - self.TSLOffset
                    return False, currPrice

                # rollingRL triggered
                elif currPrice <= self.rollingSL:
                    return True, currPrice

                # Tighten TSLprice/rollingSL
                elif (currPrice - self.rollingSL) > self.TSLOffset:
                    self.rollingSL = currPrice - self.TSLOffset
                    return False, currPrice
                else:
                    return False, currPrice

            elif side == 'Short':
                # SET rollingSL
                if firstRun == True:
                    self.rollingSL = currPrice + self.TSLOffset
                    return False, currPrice

                # rollingRL triggered
                elif currPrice >= self.rollingSL:
                    return True, currPrice

                # Tighten TSLprice/rollingSL
                elif (self.rollingSL - currPrice) > self.TSLOffset:
                    self.rollingSL = currPrice + self.TSLOffset
                    return False, currPrice
                else:
                    return False, currPrice
            else:
                print ('Bad trade side value given')
                exit()

                #test = Bitmex()


class BitmexTestv2:
    def __init__(self, VolThresh, dir, mode, loadPrevDF, TSLOffset):
        self.VolThresh = VolThresh
        self.rollingSL = 0
        self.runningTot = 0
        self.TSLOffset = TSLOffset
        self.VolThreshIterationVals = []
        self.DATA_VOL = pd.DataFrame()
        self.DATA_VOL_NORM = pd.DataFrame()
        self.DATA_ACTUAL_PRICE_VOL = pd.DataFrame()
        self.DATA_RAW = pd.DataFrame()
        self.DATA_RAW_NORM = pd.DataFrame()
        self.DATA_ACTUAL_PRICE_RAW = pd.DataFrame()
        self.NORMOBJ = NORM(MODE='STD')
        self.setupNormalise(dir, mode, loadPrevDF)
        with open(dir + '/OrderBook.txt', 'r') as f:
            self.lines = f.readlines()

        line = self.lines[0].rstrip()

        json_data = ast.literal_eval(line)
        jsonLine = json.dumps(json_data)
        jsonLine = json.loads(jsonLine)
        self.curr_ts = datetime.strptime(jsonLine[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
        self.marker = 0
        if mode == 'training':
            self.i = 0
            self.end = len(self.lines) / 2
        else:
            self.i = len(self.lines) / 2
            self.end = len(self.lines)
        return


    def setupNormalise(self, dir, mode,lag, loadPrevDF):
        print('setupNormalise')
        BMX = Bitmex()
        #lag = 200
        #cols = ['spread', 'chgMidP', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5', 'bidS_1','askS_1']
        #['bidP', 'askP']
        DATA, DATA_ACTUAL_PRICE = BMX.collectDatav3(dir, lag, mode, loadPrevDF)
        self.DATA_RAW = DATA
        self.DATA_ACTUAL_PRICE_RAW = DATA_ACTUAL_PRICE



        # DATA_ACTUAL_PRICE_before_reduction = DATA_ACTUAL_PRICE

        newNpData = []
        newNpDataIndexes = []
        newNpDataActual = []
        newNpDataIndexesActual = []
        npData = DATA.as_matrix()
        npDataActual = DATA_ACTUAL_PRICE.as_matrix()
        indexVals = DATA.index.values
        indexValsActual = DATA_ACTUAL_PRICE.index.values
        entryPriceLong = []
        entryPriceShort = []
        exitPriceLong = []
        exitPriceShort = []
        entryIndex = []
        exitIndexLong = []
        exitIndexShort = []
        for x in range(0, len(npData)):
            if npData[x][7] > self.VolThresh or npData[x][8] > self.VolThresh:
                newNpData.append(npData[x])
                newNpDataIndexes.append(indexVals[x])
                newNpDataIndexesActual.append(indexValsActual[x])
                self.VolThreshIterationVals.append(x)
                buff1 = self.getExitLong(DATA_ACTUAL_PRICE,x)
                buff2 = self.getExitShort(DATA_ACTUAL_PRICE, x)

                buff = [buff1,buff2]
                newNpDataActual.append(buff)

        npData = np.asarray(newNpData, dtype=float)
        df_Out = pd.DataFrame(data=npData, columns=DATA.columns, index=newNpDataIndexes, dtype=np.float32)
        df_OutActual = pd.DataFrame(data=newNpDataActual, columns=['tradeinfo'],
                                    index=newNpDataIndexesActual, dtype=np.float32)
        # df_Out
        DATA = df_Out
        DATA_ACTUAL_PRICE = df_OutActual.get_value
        self.DATA_VOL = DATA
        # [side, entryIndex, exitIndex, entryPrice, exitPrice, -1.00 * pcentPriceChgShort]

        self.DATA_ACTUAL_PRICE_VOL = DATA_ACTUAL_PRICE

        print('***SETUP NORMALISATION***')
        print('len(DATA_ACTUAL_PRICE): ' + str(len(DATA_ACTUAL_PRICE)))
        print('len(DATA): ' + str(len(DATA)))
        print('self.VolThreshIterationVals: ' + str(self.VolThreshIterationVals))
        DATA = DATA.dropna()
        self.DATA_RAW_NORM = self.NORMOBJ.IN(self.DATA_RAW, isLog=False)
        self.DATA_VOL_NORM = self.NORMOBJ.IN(DATA, isLog=False)

    def getExitLong(self,DATA,x):
        print ('getExitLong')
        bid = DATA.get_value(x,col='bidP')
        ask = DATA.get_value(x, col='askP')
        # ask
        side = 'Long'
        firstRun = True
        isTSLTrigged = False
        currPrice = ask
        entryPrice = currPrice
        entryIndex = x
        prevBid = 0
        while isTSLTrigged == False:
            isTSLTrigged, slPrice = self.isTSLTriggered(side, firstRun, currPrice)

            if firstRun == False:
                while prevBid == bid:
                    bid = DATA.get_value(x, col='bidP')
                    ask = DATA.get_value(x, col='askP')

            else:
                bid = DATA.get_value(x, col='bidP')
                ask = DATA.get_value(x, col='askP')

            currPrice = bid
            prevBid = bid
            firstRun = False
            x = x + 1

        exitPrice = self.rollingSL
        exitIndex = x
        # price percentage change
        pcentPriceChgLong = ((exitPrice - entryPrice) / entryPrice) * 100
        # pcentPriceChgShort = ((minExitPriceShort - entryPriceShort) / entryPriceShort) * 100

        # reward metric
        actions = [side, entryIndex, exitIndex, entryPrice, exitPrice, pcentPriceChgLong]
        return actions

    def getExitShort(self,DATA,x):
        print ('getExitShort')

        bid = DATA.get_value(x,col='bidP')
        ask = DATA.get_value(x, col='askP')# ask
        side = 'Short'
        firstRun = True
        isTSLTrigged = False
        currPrice = bid
        entryPrice = currPrice
        entryIndex = x
        prevAsk = 0
        while isTSLTrigged == False:
            isTSLTrigged, slPrice = self.isTSLTriggered(side, firstRun, currPrice)
            if firstRun == False:
                while prevAsk == ask:
                    bid = DATA.get_value(x, col='bidP')
                    ask = DATA.get_value(x, col='askP')# prevAsk = ask
            else:

                bid = DATA.get_value(x, col='bidP')
                ask = DATA.get_value(x, col='askP')

            currPrice = ask
            prevAsk = ask

            firstRun = False
            x = x + 1

        exitPrice = self.rollingSL
        exitIndex = x
        # price percentage change
        pcentPriceChgShort = ((exitPrice - entryPrice) / entryPrice) * 100
        # pcentPriceChgShort = ((minExitPriceShort - entryPriceShort) / entryPriceShort) * 100

        # reward metric
        actions = [side, entryIndex, exitIndex, entryPrice, exitPrice, -1.00 * pcentPriceChgShort]

        return actions

    def goLong(self):
        print ('goLong')
        [buffLong, buffShort] = self.DATA_ACTUAL_PRICE_VOL.get_value(index=self.marker,col='tradeinfo')
        #buffLong = [side, entryIndex, exitIndex, entryPrice, exitPrice, -1.00 * pcentPriceChgShort]

        self.marker = self.marker + 1
        return buffLong
    def goShort(self):
        print ('goShort')
        [buffLong, buffShort] = self.DATA_ACTUAL_PRICE_VOL.get_value(index=self.marker, col='tradeinfo')
        # buffLong = [side, entryIndex, exitIndex, entryPrice, exitPrice, -1.00 * pcentPriceChgShort]

        self.marker = self.marker + 1
        return buffShort

    def reset(self):
        print ('reset')

        self.marker = 0

    def get_obs(self):
        print ('get_obs')
        #['spread', 'chgMidP', 'OBI_LVL_1', 'OBI_LVL_2', 'OBI_LVL_3', 'OBI_LVL_4', 'OBI_LVL_5', 'bidS_1','askS_1']
        spread = self.DATA_VOL_NORM.get_value(index=self.marker, col='spread')
        chgMidP = self.DATA_VOL_NORM.get_value(index=self.marker, col='chgMidP')
        OBI_LVL_1 = self.DATA_VOL_NORM.get_value(index=self.marker, col='OBI_LVL_1')
        OBI_LVL_2 = self.DATA_VOL_NORM.get_value(index=self.marker, col='OBI_LVL_2')
        OBI_LVL_3 = self.DATA_VOL_NORM.get_value(index=self.marker, col='OBI_LVL_3')
        OBI_LVL_4 = self.DATA_VOL_NORM.get_value(index=self.marker, col='OBI_LVL_4')
        OBI_LVL_5 = self.DATA_VOL_NORM.get_value(index=self.marker, col='OBI_LVL_5')
        bidS_1 = self.DATA_VOL_NORM.get_value(index=self.marker, col='bidS_1')
        askS_1 = self.DATA_VOL_NORM.get_value(index=self.marker, col='askS_1')
        return [spread, chgMidP, OBI_LVL_1, OBI_LVL_2, OBI_LVL_3, OBI_LVL_4, OBI_LVL_5, bidS_1,askS_1]
    def isTSLTriggered(self, side, firstRun, currPrice):

        # print ('isTSLTriggered')
        if self.TSLOffset == 0:
            print('Bad offset value given.')
            exit()
        elif self.TSLOffset < 0:
            self.TSLOffset = -1.00 * self.TSLOffset

        if side == 'Long':
            # SET rollingSL
            if firstRun == True:
                self.rollingSL = currPrice - self.TSLOffset
                return False, currPrice

            # rollingRL triggered
            elif currPrice <= self.rollingSL:
                return True, currPrice

            # Tighten TSLprice/rollingSL
            elif (currPrice - self.rollingSL) > self.TSLOffset:
                self.rollingSL = currPrice - self.TSLOffset
                return False, currPrice
            else:
                return False, currPrice

        elif side == 'Short':
            # SET rollingSL
            if firstRun == True:
                self.rollingSL = currPrice + self.TSLOffset
                return False, currPrice

            # rollingRL triggered
            elif currPrice >= self.rollingSL:
                return True, currPrice

            # Tighten TSLprice/rollingSL
            elif (self.rollingSL - currPrice) > self.TSLOffset:
                self.rollingSL = currPrice + self.TSLOffset
                return False, currPrice
            else:
                return False, currPrice
        else:
            print ('Bad trade side value given')
            exit()

            # test = Bitmex()