import numpy as np
import pandas as pd
from core import DataGenerator
from DataNormalisation import DataNormalisation as NORM

class WavySignal(DataGenerator):
    """Modulated sine generator
    """
    @staticmethod
    def _generator(period_1, period_2, epsilon, ba_spread=0):
        i = 0
        while True:
            i += 1
            bid_price = (1 - epsilon) * np.sin(2 * i * np.pi / period_1) + \
                epsilon * np.sin(2 * i * np.pi / period_2)
            yield bid_price, bid_price + ba_spread

class WavySignal2(DataGenerator):
    @staticmethod
    def _generator(period_1, period_2, epsilon, ba_spread=0):
        i = 0
        while True:
            i += 1
            bid_price = (1 - epsilon) * np.sin(2 * i * np.pi / period_1) + \
                epsilon * np.sin(2 * i * np.pi / period_2)
            yield bid_price, bid_price + ba_spread, bid_price, bid_price + ba_spread


class myGen(DataGenerator):

    @staticmethod
    def _generator(path,resamp_rate,MODE):
        NM = NORM(MODE=MODE)
        with open(path, 'rb') as f:
            df = pd.read_csv(f, index_col=1, parse_dates=True)
            df.drop(df.columns[[0]], axis=1, inplace=True)
            df = df.resample(resamp_rate, how='ohlc', axis=0)
            df = df.dropna()
            df = NM.IN(df)
            arr = df.as_matrix()

            for x in range(0, len(arr), 1):
                #print(str(arr[x][0]),str(arr[x][4]))  # open BID
                #BO,BH, BL, BC, AO, AH, AL, AC
                #0, 1,  2,  3,  4,  5,  6,  7
                yield float(arr[x][0]), float(arr[x][4])

class myGen_10T_DIFF(DataGenerator):

    @staticmethod
    def _generator():
        with open('AUDUSD-2017-07_10T_DIFF.csv', 'rb') as f:
            df = pd.read_csv(f, index_col=1, parse_dates=True)
            arr = df.as_matrix()

            for x in range(0, len(arr), 1):
                # print(str(arr[x][0]),str(arr[x][4]))  # open BID
                # BO,BH, BL, BC, AO, AH, AL, AC
                # 0, 1,  2,  3,  4,  5,  6,  7
                yield float(arr[x][0]), float(arr[x][4])

class myGen_10T_STD(DataGenerator):
    @staticmethod
    def _generator():
        with open('AUDUSD-2017-07_10T_STD.csv', 'rb') as f:
            df = pd.read_csv(f, index_col=1, parse_dates=True)
            arr = df.as_matrix()

            for x in range(0, len(arr), 1):
                # print(str(arr[x][0]),str(arr[x][4]))  # open BID
                # BO,BH, BL, BC, AO, AH, AL, AC
                # 0, 1,  2,  3,  4,  5,  6,  7
                yield float(arr[x][0]), float(arr[x][4])

class myGen_File(DataGenerator):
    @staticmethod
    def _generator(path):
        with open(path, 'rb') as f:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            arr = df.as_matrix()
            arr = arr[2:]
            #print (arr[2:])

            for x in range(0, len(arr), 1):
                #print(str(arr[x][0]),str(arr[x][4]))  # open BID
                # BO,BH, BL, BC, AO, AH, AL, AC
                # 0, 1,  2,  3,  4,  5,  6,  7
                yield float(arr[x][0]), float(arr[x][4])

class myGen_DF(DataGenerator):

    @staticmethod
    def _generator(df):
        arr = df.as_matrix()
        rows,cols = arr.shape

        #print('_generator')
        for r in range(0, rows, 1):
            buff = []
            for c in range(0, cols, 1):
                buff.append(float(arr[r][c]))
            yield buff
class myGen_npData(DataGenerator):

    @staticmethod
    def _generator(npData):
        for r in range(0, len(npData), 1):
            yield npData[r]


class myGenDiff(DataGenerator):
    @staticmethod
    def _generator():
        NM = NORM(MODE='DIFF')
        with open('./AUDUSD-2017-07.csv', 'rb') as f:
            df = pd.read_csv(f, index_col=1, parse_dates=True)
            df.drop(df.columns[[0]], axis=1, inplace=True)
            df = df.resample('1H', how='ohlc', axis=0)
            df = df.dropna()
            df = NM.IN(df)
            arr = df.as_matrix()

            for x in range(0, len(arr), 1):
                #print(str(arr[x][0]),str(arr[x][4]))  # open BID
                yield float(arr[x][0]), float(arr[x][4])


class myGen2(DataGenerator):
    @staticmethod
    def _generator(avg):
        bid_price = 0.0
        ask_price = 0.0
        with open("./data.txt") as f:
            for line in f:
                bid_price = 0.0
                ask_price = 0.0
                for _ in range(avg):
                    bid, ask = line.rstrip('\n').rstrip('\r').split(',')
                    bid_price = bid_price + float(bid)
                    ask_price = ask_price + float(ask)
                bid_price = float(bid_price) / float(avg)
                ask_price = float(ask_price) / float(avg)
                yield float(bid_price), float(ask_price)

