# !/usr/bin/env python
import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests as r
from datetime import datetime, timedelta
import time

import datetime
from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient
from BitMEXAPIKeyAuthenticator import APIKeyAuthenticator

from DataNormalisation import DataNormalisation as NORM
import pprint
import time

class BITMEX_API:

    def __init__(self):
        self.bitMEXAuth = ''
        self.pp = pprint.PrettyPrinter(indent=2)
        self.verbose = False
        self.oldEpoch = time.time() - 60
        '''
        HOST = "https://testnet.bitmex.com"
        API_KEY = 'xj7j0ELpTpbMpAR0pNgo5hC4'
        API_SECRET = 'S_b6_1cHxqe0EJJa1ORWzAo5R-bQcYvYjCL45lP3539QwbW1'

        '''
        HOST = 'https://www.bitmex.com'
        API_KEY = '90D7yvSWtLfNz0AR3ePGcsVi'
        API_SECRET = 'fQgZTvJQlT7Ix1hb3e6bkEANewLroEAHIom7VgCHGOKdaUo8'
        self.initialise_API(HOST=HOST, API_KEY=API_KEY, API_SECRET=API_SECRET)

        #print(dir(self.bitMEX))
        #print(dir(self.bitMEXAuth))
        #print('self.bitMEXAuth.Execution: ' + str(dir(self.bitMEXAuth.Execution)))
        #print('self.bitMEXAuth.Funding: ' + str(dir(self.bitMEXAuth.Funding)))
        #print('self.bitMEXAuth.Order: ' + str(dir(self.bitMEXAuth.Order)))
        #print('self.bitMEXAuth.Position: ' + str(dir(self.bitMEXAuth.Position)))
        #print('self.bitMEXAuth.Stats: ' + str(dir(self.bitMEXAuth.Stats)))
        #print('self.bitMEXAuth.Trade: ' + str(dir(self.bitMEXAuth.Trade)))
        #print(dir(bitMEXAuthenticated.Position))
        return

    def initialise_API(self,HOST,API_KEY,API_SECRET):
        SPEC_URI = HOST + "/api/explorer/swagger.json"

        config = {
            # Don't use models (Python classes) instead of dicts for #/definitions/{models}
            'use_models': False,
            # This library has some issues with nullable fields
            'validate_responses': False,
            # Returns response in 2-tuple of (body, response); if False, will only return body
            'also_return_response': True,
        }       # See full config options at http://bravado.readthedocs.io/en/latest/configuration.html
        request_client = RequestsClient()
        request_client.authenticator = APIKeyAuthenticator(HOST, API_KEY, API_SECRET)
        self.bitMEXAuth = SwaggerClient.from_url(SPEC_URI, config=config, http_client=request_client)
        time.sleep(2)
    def printAPIINFO(self):

        print(dir(self.bitMEXAuth))
        print('self.bitMEXAuth.Execution: ' + str(dir(self.bitMEXAuth.Execution)))
        print('self.bitMEXAuth.Funding: ' + str(dir(self.bitMEXAuth.Funding)))
        print('self.bitMEXAuth.Order: ' + str(dir(self.bitMEXAuth.Order)))
        print('self.bitMEXAuth.Position: ' + str(dir(self.bitMEXAuth.Position)))
        print('self.bitMEXAuth.Stats: ' + str(dir(self.bitMEXAuth.Stats)))
        print('self.bitMEXAuth.Trade: ' + str(dir(self.bitMEXAuth.Trade)))

    def getTrade(self):
        print('Get Trade')
        # authenticated call
        res, http_response = self.bitMEXAuth.Trade.Trade_get(symbol='XBTUSD', count=1).result()
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res

    def getOpenOrders(self,symbol,count):
        # BUY
        # price Of order placed on orderbook. once market price hits limitorder price the trade is placed
        print('get Open Orders')
        # Basic order placement
        # print(dir(bitMEXAuthenticated.Order))
        # displayQty=int(0),
        # check takeProfitPrice logic
        try:
            res, http_response = self.bitMEXAuth.Order.Order_getOrders(symbol=symbol,
                                                             filter='{"open": true}',
                                                             count=count).result()  # ,,pegPriceType='TrailingStopPeg',pegOffsetValue=-5.00stopPx=stopPx,price=CURR_PRICE,execInst='LastPrice'
        except:
            return -1
        if self.verbose == True:

            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res

    def getExecutionHist(self,symbol,count):
        print('Get Execution History')
        # authenticated call
        res, http_response = self.bitMEXAuth.Execution.Execution_getTradeHistory(symbol=symbol, count=count,reverse=True).result()
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res

    def getPosition(self,symbol):
        # Basic authenticated call
        print('Get Position')
        res, http_response = self.bitMEXAuth.Position.Position_get(filter=json.dumps({'symbol': symbol})).result()
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)
        #arr = json.loads(jsonData)
        #unrealisedRoePcnt = arr[0]

        return res
    def getOpenPositions(self,symbol):
        # Basic authenticated call
        print('Get Open Positions')
        res, http_response = self.bitMEXAuth.Position.Position_get(filter='{"isOpen": true}').result()
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)
        #arr = json.loads(jsonData)
        #unrealisedRoePcnt = arr[0]

        return res
    def getUnrealisedROEPercent(self):
        # Basic authenticated call
        print('Get Unrealised ROE Percent')
        res, http_response = self.bitMEXAuth.Position.Position_get(filter=json.dumps({'symbol': 'XBTUSD'})).result()
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)
        # arr = json.loads(jsonData)
        # unrealisedRoePcnt = arr[0]

        return res[0]['unrealisedRoePcnt']

    def updateLeverage(self,symbol,lev):
        print('Update Position Leverage')
        #LEV = number between 0.01 and 100 to enable isolated margin with a fixed leverage. Send 0 to enable cross
        res, http_response = self.bitMEXAuth.Position.Position_updateLeverage(symbol=symbol,leverage=lev).result()
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res

    def createLimitOrderImmediate(self,symbol,simpleOrderQty,side,price):
        #BUY
        #price Of order placed on orderbook. once market price hits limitorder price the trade is placed
        print('Create Immediate Limit Order')
        # Basic order placement
        # print(dir(bitMEXAuthenticated.Order))
        #displayQty=int(0),
        #check takeProfitPrice logic
        res, http_response = self.bitMEXAuth.Order.Order_new(symbol=symbol,side=side, simpleOrderQty=simpleOrderQty,price=price,
                                                          ordType='Limit',timeInForce='ImmediateOrCancel').result()#,,pegPriceType='TrailingStopPeg',pegOffsetValue=-5.00stopPx=stopPx,price=CURR_PRICE,execInst='LastPrice'
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res

    def createLimitOrder(self,symbol,simpleOrderQty,side,price):
        #BUY
        #price Of order placed on orderbook. once market price hits limitorder price the trade is placed
        print('Create Limit Order')
        # Basic order placement
        # print(dir(bitMEXAuthenticated.Order))
        #displayQty=int(0),
        #check takeProfitPrice logic
        res, http_response = self.bitMEXAuth.Order.Order_new(symbol=symbol,side=side, simpleOrderQty=simpleOrderQty,price=price,
                                                          ordType='Limit',timeInForce='GoodTillCancel').result()#,,pegPriceType='TrailingStopPeg',pegOffsetValue=-5.00stopPx=stopPx,price=CURR_PRICE,execInst='LastPrice'
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res

    def createLimitIfTouchedOrder(self,symbol,simpleOrderQty,side,price,pegOffsetValue):
        #BUY
        #price Of order placed on orderbook. once market price hits limitorder price the trade is placed
        print('create Limit If Touched Order')
        # Basic order placement
        # print(dir(bitMEXAuthenticated.Order))
        #displayQty=int(0),
        #check takeProfitPrice logic
        res, http_response = self.bitMEXAuth.Order.Order_new(symbol=symbol,side=side, simpleOrderQty=simpleOrderQty,price=price,
                                                            ordType='LimitIfTouched',timeInForce='GoodTillCancel',
                                                            pegPriceType='TrailingStopPeg',
                                                            pegOffsetValue=pegOffsetValue,execInst='LastPrice').result()#,,pegPriceType='TrailingStopPeg',pegOffsetValue=-5.00stopPx=stopPx,price=CURR_PRICE,execInst='LastPrice'
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res
    def createStopLimitOrder(self,symbol,simpleOrderQty,side,price,pegOffsetValue):
        #BUY
        #price Of order placed on orderbook. once market price hits limitorder price the trade is placed
        #when current Market price hits price or trailing stop a revese sell order is executed and put onto the orderbook cancelling our current buy order.
        print('create Stop Limit Order')



        # Basic order placement
        # print(dir(bitMEXAuthenticated.Order))
        #displayQty=int(0),
        #check takeProfitPrice logic
        res, http_response = self.bitMEXAuth.Order.Order_new(symbol=symbol,side=side, simpleOrderQty=simpleOrderQty,
                                                            ordType='StopLimit',timeInForce='FillOrKill',
                                                            pegPriceType='TrailingStopPeg',price=price,
                                                            pegOffsetValue=pegOffsetValue,execInst='LastPrice').result()#,,pegPriceType='TrailingStopPeg',pegOffsetValue=-5.00stopPx=stopPx,price=CURR_PRICE,execInst='LastPrice'
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res

    def createStopMarketOrder(self, symbol, simpleOrderQty, side, price, pegOffsetValue):
        # BUY
        # price Of order placed on orderbook. once market price hits limitorder price the trade is placed
        # when current Market price hits price or trailing stop a revese sell order is executed and put onto the orderbook cancelling our current buy order.
        print('create Stop Market Order')

        # Basic order placement
        # print(dir(bitMEXAuthenticated.Order))
        # displayQty=int(0),
        # check takeProfitPrice logic
        res, http_response = self.bitMEXAuth.Order.Order_new(symbol=symbol, side=side, simpleOrderQty=simpleOrderQty,
                                                             ordType='Stop', timeInForce='ImmediateOrCancel',
                                                             pegPriceType='TrailingStopPeg', price=price,
                                                             pegOffsetValue=pegOffsetValue,
                                                             execInst='LastPrice').result()  # ,,pegPriceType='TrailingStopPeg',pegOffsetValue=-5.00stopPx=stopPx,price=CURR_PRICE,execInst='LastPrice'
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res

    def createTrailingStop(self,symbol,simpleOrderQty,trailPrice):
        print('Create TrailingStop')
        #trailPrice is is distance from the current price at which the trailing stop will trigger.eg: if 2 if the ucrrent price moves 2 against your posiition the stop will activate.
        #neg trailPrice to stop Longs
        # pos trailPrice to stop shorts
        #Requires `ordType`: `'Stop', 'StopLimit', 'MarketIfTouched', 'LimitIfTouched'`.\n\n#### Simple Quantities\n\nSend a `simpleOrderQty` instead of an `orderQty` to create an order denominated in the underlying currency.\nThis is useful for opening up a position with 1 XBT of exposure without having to calculate how many contracts it is.
        res, http_response = self.bitMEXAuth.Order.Order_new(symbol=symbol, simpleOrderQty=simpleOrderQty,
                                                             ordType='Stop',pegPriceType='TrailingStopPeg',
                                                             pegOffsetValue=trailPrice,execInst='LastPrice',
                                                             timeInForce='GoodTillCancel'
                                                            ).result()  # ,stopPx=stopPx,price=CURR_PRICE   execInst='LastPrice'
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res

    def getOrderBook(self,symbol):
        print('Create "Limit Order"')
        # Basic order placement
        # print(dir(bitMEXAuthenticated.Order))
        # displayQty=int(0),
        # check takeProfitPrice logic
        res, http_response = self.bitMEXAuth.OrderBook.OrderBook_get(symbol=symbol, depth=1).result()  # ,stopPx=stopPx,price=CURR_PRICE
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res

    def closePositions(self,symbol):
        print('Close All Positions')
        # Basic order placement
        # print(dir(bitMEXAuthenticated.Order))
        #displayQty=int(0),
        res, http_response = self.bitMEXAuth.Order.Order_closePosition(symbol=symbol).result()
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res

    def closeAllOrders(self):
        print('Close All Orders')
        # Basic order placement
        # print(dir(bitMEXAuthenticated.Order))
        #displayQty=int(0),
        res, http_response = self.bitMEXAuth.Order.Order_cancelAll(symbol='XBTUSD').result()
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res

    def closeOrder(self,orderID):
        #Order_cancel
        print('Close an Order')
        # Basic order placement
        # print(dir(bitMEXAuthenticated.Order))
        # displayQty=int(0),
        res, http_response = self.bitMEXAuth.Order.Order_cancel(orderID=orderID).result()
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res
    def closeAllOrdersAfter(self, ms_timeout):
        print('Dead man switch incase of power outage.')
        #set to 0ms ot cancel this timer
        res, http_response = self.bitMEXAuth.Order.Order_cancelAllAfter(timeout=str(ms_timeout)).result()
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res

    def getStats(self):
        print('get Stats')
        # Basic order placement
        # print(dir(bitMEXAuthenticated.Order))
        #displayQty=int(0),
        res, http_response = self.bitMEXAuth.Stats.Stats_get().result()
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return res

    def getAvailableMargin(self):
        print('get AVAIL MARGIN')
        # Basic order placement
        # print(dir(bitMEXAuthenticated.Order))
        #displayQty=int(0),
        res, http_response = self.bitMEXAuth.User.User_getMargin(currency='XBt').result()
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)

        return float(res['availableMargin'])/100000000.00

    def getMinimumWithdrawalFee(self):
        print('get Minimum Withdrawal Fee')
        res, http_response = self.bitMEXAuth.User.User_minWithdrawalFee(currency='XBt').result()
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)
        return float(res['minFee'])/100000000.00

    def requestWithdrawal(self,amount,destAddr,minimumFee):
        print('request A Withdrawal')
        res, http_response = self.bitMEXAuth.User.User_requestWithdrawal(currency='XBt',amount=amount*100000000.00,address=destAddr,fee=minimumFee*100000000.00).result() #amount int64
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)
        return res


    def getDepositAddress(self):
        print('get Deposit Address')
        res, http_response = self.bitMEXAuth.User.User_getDepositAddress(currency='XBt').result()
        if self.verbose == True:
            print("Status Code: %d, headers: %s" % (http_response.status_code, http_response.headers))
            self.pp.pprint(res)
        return res

    def getLatest1mDataBitmex(self,window,delay=2):

        url = 'https://www.bitmex.com/api/v1/quote/bucketed?' \
              'binSize=1m' \
              '&partial=false' \
              '&symbol=XBT' \
              '&count='+str(window+5)+ \
              '&reverse=true'
        print('url: ' + str(url))
        response = r.get(url)
        print('status_code: ' + str(response.status_code))
        j = response.json()
        # print(j)

        indexlst = []
        lstAll = []
        for p in j:
            lst = []
            ts = datetime.datetime.strptime(p['timestamp'][0:-1], "%Y-%m-%dT%H:%M:%S.%f")
            #print ('ts: ' +str(ts))
            indexlst.append(ts)
            lst.append(p['bidPrice'])
            lst.append(p['askPrice'])
            lst.append(p['bidSize'])
            lst.append(p['askSize'])
            lstAll.append(lst)
        time.sleep(delay)

        npORIG = np.asarray(lstAll, dtype=float)
        npORIG = npORIG[::-1]
        indexlst = indexlst[::-1]
        # print(npORIG)


        cols = ['bidPrice', 'askPrice']
        colsV = ['bidSize', 'askSize']
        npORIGV = npORIG[:, 2:4]
        npORIG = npORIG[:, 0:2]
        # print(npORIG)
        # print(npORIGV)

        dfORIG = pd.DataFrame(npORIG, columns=cols, dtype=float, index=indexlst)  #
        dfORIG.index.name = 'time'
        dfORIGV = pd.DataFrame(npORIGV, columns=colsV, dtype=float, index=indexlst)  #
        dfORIGV.index.name = 'time'

        rollingAvg = dfORIG.rolling(window=window, center=False).mean()
        rollingStdDev = dfORIG.rolling(window=window, center=False).std()

        return rollingAvg.as_matrix()[-1], rollingStdDev.as_matrix()[-1]


    def minPassed(self,oldEpoch):

        newEpoch = time.time()
        if newEpoch - oldEpoch >= 60:
            return True,newEpoch
        else:
            return False,0

    def calcZSCORE(self,bidPrice):

        boolean, newEpoch = self.minPassed(self.oldEpoch)
        if boolean == True:
            self.oldEpoch = newEpoch
            #retrieve new STDDEV AND MEAN VALUES
            rollingAvg, rollingStdDev = self.getLatest1mDataBitmex(15)
            self.rollingAvg = rollingAvg
            self.rollingStdDev = rollingStdDev
            print('rollingStdDev: + ' + str(self.rollingStdDev))
            print('rollingAvg: + ' + str(self.rollingAvg))
            zscore = (bidPrice- self.rollingAvg[0])/self.rollingStdDev[0]
        else:
            zscore = (bidPrice - self.rollingAvg[0]) / self.rollingStdDev[0]

        print('zscore: ' + str(zscore))
        return zscore


#HOST                 = "https://testnet.bitmex.com"              #https://testnet.bitmex.com
#API_KEY= '5sHucwjfjgn-9NnnKvQJ2jeb'
#API_SECRET= '88ovTNXJqef655BZX5Vb7K6vNv1mBXU06fD94nLeqRsqDnAm'

BUY = BITMEX_API()
HOST = 'https://www.bitmex.com'
API_KEY = '90D7yvSWtLfNz0AR3ePGcsVi'
API_SECRET = 'fQgZTvJQlT7Ix1hb3e6bkEANewLroEAHIom7VgCHGOKdaUo8'
BUY.initialise_API(HOST=HOST, API_KEY=API_KEY, API_SECRET=API_SECRET)
'''
res = BUY.getExecutionHist('XBTUSD',2)
StopLossOrderFilledPrice = res[0]['lastPx']
StopLossTriggPrice = res[0]['stopPx']
StopLossOrder_exitTime = res[0]['transactTime']
#TSL_triggered = res[0]['triggered']
TSL_timeInForce = res[0]['timeInForce']


MarketOrder_entryTime = res[1]['transactTime']
MarketOrder_timeInForce = res[1]['timeInForce']
MarketOrder_entryPrice = res[1]['price']
print (str(MarketOrder_timeInForce) + str('::') +
str(TSL_timeInForce) + str('::') +
str(MarketOrder_entryTime) + str('::') +
str(StopLossOrder_exitTime) + str('::') +
str(MarketOrder_entryPrice) + str('::') +
str(StopLossOrderFilledPrice) + str('::') +
str(StopLossTriggPrice) + str('::'))
#BUY.pp.pprint(res)


# print('npORIG: + ' + str(npORIG))
# print('zscore: + ' + str(zscore))


#SELL.initialise_API(HOST=HOST,API_KEY=API_KEY_SELLSIDE,API_SECRET=API_SECRET_SELLSIDE)
availMargin = BUY.getAvailableMargin()
pegOffsetValueLong = -2
CURRENT_PRICE_BID = 10000
#BUY.createLimitOrder('XBTUSD', float(availMargin)/100, 'Sell',CURRENT_PRICE_BID)


BUY.printAPIINFO()
obj = BUY.getOpenPositions('XBTUSD')
print (obj)
obj = BUY.getOpenOrders('XBTUSD',1)
print (obj)
#time.sleep(10)
#orderID = obj[0]['orderID']
#BUY.closeOrder(orderID)
exit()

TOTAL_BTC_BUYSIDE  = BUY.getAvailableMargin()
#obj = json.load(TOTAL_BTC_BUYSIDE)
BUY.updateLeverage('XBTUSD',1) #1.00-100.00
sleep(0)

print('TOTAL_BTC_BUYSIDE: ' + str(TOTAL_BTC_BUYSIDE))
simpOrderQuantity = TOTAL_BTC_BUYSIDE/1000.00
resp=BUY.getOrderBook('XBTUSD')
askPrice = resp[0]['askPrice']  #>price used when going Long
bidPrice = resp[0]['bidPrice']  #<price used when going Short
pegOffsetValue = -4 #chases price going up
print ('askPrice:' + str(askPrice))
print ('simpOrderQuantity:' + str(simpOrderQuantity))

#resp = BUY.createLimitIfTouchedOrder('XBTUSD',simpOrderQuantity,'Buy',askPrice,pegOffsetValue) #BUY 1BTC @ Market VALUE
resp = BUY.createLimitOrder('XBTUSD',simpOrderQuantity,'Buy',askPrice)
print('orderID          : ' + str(resp['orderID']))
print('pegPriceType     : ' + str(resp['pegPriceType']))
print('transactTime     : ' + str(resp['transactTime']))
print('simpleOrderQty   : ' + str(resp['simpleOrderQty']))
print('triggered        : ' + str(resp['triggered']))
print('pegOffsetValue   : ' + str(resp['pegOffsetValue']))
print('execInst         : ' + str(resp['execInst']))
BUY.createStopLimitOrder('XBTUSD',simpOrderQuantity,'Sell',askPrice+pegOffsetValue,pegOffsetValue)
#BUY.createTrailingStop('XBTUSD',simpOrderQuantity,-20.00)
sleep(1)
BUY.getPosition('XBTUSD')
exit()
'''
#minimumFee = BUY.getMinimumWithdrawalFee()
#print('MinimumWithdrawalFee: ' + str(minimumFee))
#smallAccDepAddr = SELL.getDepositAddress()
#print('DepositAddress: ' + str(smallAccDepAddr))

#out = BUY.requestWithdrawal(0.222,smallAccDepAddr,minimumFee)
#print('Withdrawal request: ' + str(out))

#BUY.confirmWithdrawal()
#get current position
'''
sleep(10)
pos = BUY.getPosition('XBTUSD')
#
print('pos: ' + str(pos))
#INFO
#avgEntryPrice,breakEvenPrice
#liquidationPrice,leverage
#markPrice,markValue
#unrealisedPnl,unrealisedRoePcnt(0.2346)
#marginCallPrice

#UPDATE LEVERAGE
#test.updateLeverage('XBTUSD',100.00) #1.00-100.00
unrealisedROEPercent = BUY.getUnrealisedROEPercent()
print('unrealisedROEPercent: ' + str(unrealisedROEPercent))
#close order
sleep(10)
closed = BUY.closeOrder('XBTUSD')
# {u'ordStatus': u'Filled', u'exDestination': u'XBME', u'text': u'Submitted via API.', u'timeInForce': u'ImmediateOrCancel', u'currency': u'USD', u'pegPriceType': u'', u'simpleLeavesQty': 0.0, u'ordRejReason': u'', u'transactTime': datetime.datetime(2017, 10, 21, 5, 45, 29, 244000, tzinfo=tzutc()), u'clOrdID': u'', u'settlCurrency': u'XBt', u'displayQty': None, u'avgPx': 6077.9, u'symbol': u'XBTUSD', u'simpleOrderQty': None, u'ordType': u'Market', u'triggered': u'', u'timestamp': datetime.datetime(2017, 10, 21, 5, 45, 29, 244000, tzinfo=tzutc()), u'price': 6078.0, u'workingIndicator': False, u'pegOffsetValue': None, u'execInst': u'Close', u'simpleCumQty': 0.2567147, u'orderID': u'c868a02e-cb98-cf26-375e-347527a53868', u'multiLegReportingType': u'SingleSecurity', u'account': 6666L, u'stopPx': None, u'leavesQty': 0L, u'orderQty': 1562L, u'cumQty': 1562L, u'contingencyType': u'', u'clOrdLinkID': u'', u'side': u'Sell'}

#print('closed: ' + str(closed))

#detect SL or MC

#u'triggered': u'StopOrderTriggered',

#test.getExecutionHist('XBTUSD',1)

#test.createOrder('XBTUSD',1.00,'Buy') #BUY 1BTC @ Market VALUE

#get current position
#pos = test.getPosition()
#print(pos)
#INFO
#avgEntryPric e,breakEvenPrice
#liquidationPrice,leverage
#markPrice,markValue
#unrealisedPnl,unrealisedRoePcnt(0.2346)
#marginCallPrice

#UPDATE LEVERAGE
#test.updateLeverage('XBTUSD',100.00) #1.00-100.00
#sleep(10)
#test.updateLeverage('XBTUSD',0.00) #1.00-100.00
#sleep(10)

#close order
#test.closeOrder('XBTUSD')

#detect SL or MC

#u'triggered': u'StopOrderTriggered',

'''