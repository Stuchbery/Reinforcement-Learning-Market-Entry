import Queue as queue
import threading
import datetime
import sys
from time import sleep, time
import time
from API_BitMex import BITMEX_API

class OrderManagerLive():
    def __init__(self,VOL_THRESH):
        # INIT VALUES
        self.percentageToTrade = 5
        self.symbol = 'XBTUSD'
        self.leverageLevel = 1.00
        self.pegOffsetValueLong = -4 #LONG OFFSETVAL default
        self.pegOffsetValueShort = 4  # LONG OFFSETVAL default
        self.limitOrderToFillTimeLimit = 60 # seconds
        self.percentageFeeOffset = 0.3

        self.BID_SIZE_THRESH = VOL_THRESH
        self.ASK_SIZE_THRESH = VOL_THRESH
        self.LIMIT_PRICE_LVL = 1
        # INIT API
        self.BM_API = BITMEX_API()
        self.closeOpenPositions()

        self.totMargin = 0
        self.availMargin = 0
        availMargin = self.BM_API.getAvailableMargin()
        self.setAvailMargin(availMargin)
        self.transact_History_File = 'PnL_Log.txt'

        with open(self.transact_History_File, 'w') as f:
            f.write('TRADE_SIDE::TRADE_STATUS::%OF_ACCOUNT_TO_RISK::LEVERAGE_MULTI::TSL_OFFSET_VALUE::VOL_THRESH::OB_PRICE_LVL::'
                    'MARKET_ORDER_TIMEINFORCE::STOPLOSS_ORDER_TIMEINFORCE::MARKET_ORDER_ENTRYTIME::STOPLOSS_ORDER_EXITTIME::'
                    'MARKET_ORDER_ENTRY_PRICE::STOPLOSS_ORDER_FILLED_PRICE::STOPLOSS_ORDER_TRIGGER_PRICE::AVAILABLE_MARGIN_INITIAL::AVAILABLE_MARGIN_FINAL' + '\n')

    #SETTERS
    def setLeverageLevel(self, leverageLevel):
        self.leverageLevel = leverageLevel
        self.BM_API.updateLeverage(self.symbol, self.leverageLevel)
        return
    def setPercentageToTrade(self,percentageToTrade):
        self.percentageToTrade = percentageToTrade
        return
    def setAvailMargin(self, availMargin):
        self.totMargin = availMargin
        self.availMargin = (float(availMargin)/100)*self.percentageToTrade
        return
    def setTrailingOffsetValue(self, pegOffsetValue):
        if pegOffsetValue > 0:
            self.pegOffsetValueShort = pegOffsetValue
            self.pegOffsetValueLong = -1 * pegOffsetValue
        elif pegOffsetValue < 0:
            self.pegOffsetValueShort = -1 * pegOffsetValue
            self.pegOffsetValueLong = pegOffsetValue
        else:
            # default values
            self.pegOffsetValueLong = -4  # LONG OFFSETVAL default
            self.pegOffsetValueShort = 4  # LONG OFFSETVAL default

        return

    #API
    def minPassed(self, oldEpoch):

        newEpoch = time.time()
        if newEpoch - oldEpoch >= 61:
            return True, newEpoch
        else:
            return False, 0
    def goLong(self,ASK_PRICE):
        print('GO LONG')
        self.LIMIT_ENTRY_PRICE_BID  = ASK_PRICE

        print('ENTRY_PRICE_ASK: ' + str(self.LIMIT_ENTRY_PRICE_BID))
        AVAILABLE_MARGIN_INITIAL = float(self.totMargin)

        resp = self.BM_API.createLimitOrderImmediate(self.symbol, float(self.availMargin), 'Buy', self.LIMIT_ENTRY_PRICE_BID)
        print('resp: ' + str(resp))
        sleep(1)

        if resp['ordStatus'] =='Canceled':
            totMargin = self.BM_API.getAvailableMargin()
            AVAILABLE_MARGIN_FINAL = totMargin
            with open('PnL_Log.txt', 'a') as f:
                f.write('LONG' + str('::') + \
                        str('MARKET_ORDER_CANCELLED') + str('::') + \
                        str(self.percentageToTrade) + str('::') + \
                        str(self.leverageLevel) + str('::') + \
                        str(self.pegOffsetValueLong) + str('::') + \
                        str(self.BID_SIZE_THRESH) + str('::') + \
                        str(self.LIMIT_PRICE_LVL) + str('::') + \
                        str('-') + str('::') + \
                        str('-') + str('::') + \
                        str(datetime.datetime.utcnow()) + str('::') + \
                        str('-') + str('::') + \
                        str('-') + str('::') + \
                        str('-') + str('::') + \
                        str('-') + str('::') + \
                        str(AVAILABLE_MARGIN_INITIAL) + str('::') + \
                        str(AVAILABLE_MARGIN_FINAL) + \
                        '\n')
            return
        self.BM_API.createStopLimitOrder('XBTUSD', float(self.availMargin), 'Sell',self.LIMIT_ENTRY_PRICE_BID + self.pegOffsetValueLong, self.pegOffsetValueLong)

        self.waitForStopLossToTrigger()

        self.closeOpenPositions()

        totMargin        = self.BM_API.getAvailableMargin()
        AVAILABLE_MARGIN_FINAL  = totMargin
        self.setAvailMargin(totMargin)

        self.saveTradeToLogFile('LONG', AVAILABLE_MARGIN_INITIAL, AVAILABLE_MARGIN_FINAL)
        reward = self.calcRewardValue('LONG')
        return reward

    def goShort(self,BID_PRICE):
        print ('GO SHORT')
        #self.entryintobookEpoch = time.time()
        self.LIMIT_ENTRY_PRICE_BID  = BID_PRICE
        print('BID_PRICE: ' + str(self.LIMIT_ENTRY_PRICE_BID))
        AVAILABLE_MARGIN_INITIAL = float(self.totMargin)

        resp = self.BM_API.createLimitOrderImmediate(self.symbol, float(self.availMargin), 'Sell', self.LIMIT_ENTRY_PRICE_BID)
        print('resp: ' + str(resp))
        sleep(1)

        if resp['ordStatus'] =='Canceled':
            totMargin = self.BM_API.getAvailableMargin()
            AVAILABLE_MARGIN_FINAL = totMargin
            with open('PnL_Log.txt', 'a') as f:
                f.write('SHORT' + str('::') + \
                        str('MARKET_ORDER_CANCELLED') + str('::') + \
                        str(self.percentageToTrade) + str('::') + \
                        str(self.leverageLevel) + str('::') + \
                        str(self.pegOffsetValueLong) + str('::') + \
                        str(self.BID_SIZE_THRESH) + str('::') + \
                        str(self.LIMIT_PRICE_LVL) + str('::') + \
                        str('-') + str('::') + \
                        str('-') + str('::') + \
                        str(datetime.datetime.utcnow()) + str('::') + \
                        str('-') + str('::') + \
                        str('-') + str('::') + \
                        str('-') + str('::') + \
                        str('-') + str('::') + \
                        str(AVAILABLE_MARGIN_INITIAL) + str('::') + \
                        str(AVAILABLE_MARGIN_FINAL) + \
                        '\n')
            return

        self.BM_API.createStopLimitOrder('XBTUSD', float(self.availMargin), 'Buy',self.LIMIT_ENTRY_PRICE_BID + self.pegOffsetValueShort, self.pegOffsetValueShort)

        self.waitForStopLossToTrigger()

        self.closeOpenPositions()

        # update available margin in account
        totMargin = self.BM_API.getAvailableMargin()
        AVAILABLE_MARGIN_FINAL  = totMargin
        self.setAvailMargin(totMargin)

        self.saveTradeToLogFile('SHORT',AVAILABLE_MARGIN_INITIAL,AVAILABLE_MARGIN_FINAL)

        reward = self.calcRewardValue('SHORT')
        return reward

    def waitForStopLossToTrigger(self):
        sleep(1)
        # wait for trailing stop order to trigger
        SLO = self.BM_API.getOpenOrders(self.symbol, 1)
        while len(SLO) > 0:  # True = LimitOrder still not filled
            print('waitForStopLossToTrigger: ' + str(SLO))
            sleep(1.5)
            SLO = self.BM_API.getOpenOrders(self.symbol, 1)

        return True
    def closeOpenPositions(self):
        print('closeOpenPositions')
        sleep(1)
        posStatus = self.BM_API.getOpenPositions(self.symbol)
        print('posStatus: ' + str(posStatus))
        if len(posStatus) > 0:  # trailing SL couldn't fill. cancel position
            self.BM_API.closePositions(self.symbol)
            sleep(1)
    def saveTradeToLogFile(self,side,AVAILABLE_MARGIN_INITIAL,AVAILABLE_MARGIN_FINAL):
        pegOffsetVal = 0
        volThresh = 0
        if side == 'SHORT':
            pegOffsetVal = self.pegOffsetValueShort
            volThresh = self.ASK_SIZE_THRESH
        else:
            pegOffsetVal = self.pegOffsetValueLong
            volThresh = self.BID_SIZE_THRESH

        res = self.BM_API.getExecutionHist('XBTUSD', 2)
        StopLossOrderFilledPrice = res[0]['lastPx']
        StopLossTriggPrice = res[0]['stopPx']
        StopLossOrder_exitTime = res[0]['transactTime']
        TSL_timeInForce = res[0]['timeInForce']

        MarketOrder_entryTime = res[1]['transactTime']
        MarketOrder_timeInForce = res[1]['timeInForce']
        MarketOrder_entryPrice = res[1]['price']

        with open(self.transact_History_File, 'a') as f:
            f.write(str(side) + str('::') + \
                    str('TRADE_COMPLETE') + str('::') + \
                    str(self.percentageToTrade) + str('::') + \
                    str(self.leverageLevel) + str('::') + \
                    str(pegOffsetVal) + str('::') + \
                    str(volThresh) + str('::') + \
                    str(self.LIMIT_PRICE_LVL) + str('::') + \
                    str(MarketOrder_timeInForce) + str('::') + \
                    str(TSL_timeInForce) + str('::') + \
                    str(MarketOrder_entryTime) + str('::') + \
                    str(StopLossOrder_exitTime) + str('::') + \
                    str(MarketOrder_entryPrice) + str('::') + \
                    str(StopLossOrderFilledPrice) + str('::') + \
                    str(StopLossTriggPrice) + str('::') + \
                    str(AVAILABLE_MARGIN_INITIAL) + str('::') + \
                    str(AVAILABLE_MARGIN_FINAL) + \
                    '\n')
    def calcRewardValue(self,side):

        res = self.BM_API.getExecutionHist('XBTUSD', 2)


        StopLossOrderFilledPrice = res[0]['lastPx']
        MarketOrder_entryPrice = res[1]['price']

        #pcentPriceChgLong = ((maxExitPriceLong - entryPriceLong) / entryPriceLong) * 100
        #pcentPriceChgShort = ((minExitPriceShort - entryPriceShort) / entryPriceShort) * 100

        if side == 'SHORT':
            pcentPriceChgShort = (-1.00*(((float(StopLossOrderFilledPrice) - float(MarketOrder_entryPrice)) / float(MarketOrder_entryPrice)) * 100.0))  - self.percentageFeeOffset
            return pcentPriceChgShort
        else:
            pcentPriceChgLong = (((float(StopLossOrderFilledPrice) - float(MarketOrder_entryPrice)) / float(MarketOrder_entryPrice)) * 100.0) - self.percentageFeeOffset
            return pcentPriceChgLong
    def datetime_to_float(self,d):
        epoch = datetime.datetime.utcfromtimestamp(0)
        total_seconds = (d - epoch).total_seconds()
        # total_seconds will be in decimals (millisecond precision)
        return total_seconds*1000
    def float_to_datetime(self,fl):
        return datetime.datetime.fromtimestamp(fl)

#test = OrderManagerLive(1)

#test.goLong(8236.5)
#test.goShort(8236.5)