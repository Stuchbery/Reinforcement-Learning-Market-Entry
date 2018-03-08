import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
class TechnicalIndicators:
    Price = 0
    a1 = 0
    b1 = 0
    coef1 = 0
    coef2 = 0
    coef3 = 0
    Filt0 = 0
    Filt1 = 0
    Filt2 = 0
    count = 0
    prevVal = 0
    indicator = 0
    Period = 6

    def setPeriod(self,newPeriod):
        global Period
        Period=newPeriod

    def resetSS(self):
        global Price
        global a1
        global b1
        global coef1
        global coef2
        global coef3
        global Filt0
        global Filt1
        global Filt2
        global count
        global prevVal
        global indicator
        #global Period
        Price=0
        a1=0
        b1=0
        coef1=0
        coef2=0
        coef3=0
        Filt0=0
        Filt1=0
        Filt2=0
        count=0
        prevVal=0
        indicator=0
        #Period=6

    def SS(self,avgPrice):
        global Price
        global a1
        global b1
        global coef1
        global coef2
        global coef3
        global Filt0
        global Filt1
        global Filt2
        global count
        global prevVal
        global indicator
        global Period

        Price=avgPrice
        a1 = math.exp(-1.414*3.14159 / Period)
        b1 = 2*a1*math.cos(math.radians(1.414*180 / Period))
        coef2 = b1
        coef3 = -a1*a1
        coef1 = 1 - coef2 - coef3
        Filt0 = coef1*Price + coef2*Filt1 + coef3*Filt2

        if count < 3:
            Filt0 = Price
            count = count+1

        #print"----------------------"
        #print"Price:"+str(Price)
        #print"a1:"+str(a1)
        #print"b1:"+str(b1)
        #print"coef1:"+str(coef1)
        #print"coef2:"+str(coef2)
        #print"coef3:"+str(coef3)
        #print"Filt0:"+str(Filt0)
        #print"Filt1:"+str(Filt1)
        #print"Filt2:"+str(Filt2)
        #print"count:"+str(count)
        #print"prevVal:"+str(prevVal)
        #print"indicator:"+str(indicator)
        #print"Period:"+str(Period)
        #print"----------------------"

        #SSAVG
        indicator=Filt0


        #update price of next comparison
        prevVal=Filt0

        #create buy/sell indicator
        #update values for next bar
        Filt2=Filt1  #old val becomes older val
        Filt1=Filt0  #new val becomes old val

        return indicator

    #if __name__ == '__main__':

    def LP_Filter(self,df,period=6):
        npData=df.as_matrix()
        self.setPeriod(period)
        npDataSS = []
        if len(npData.shape) == 3:
            batches,samples,features = npData.shape
            npDataSS = np.zeros((batches,samples,features),dtype=float)
            for b in range(0,batches):
                for f in range(0,features):
                    self.resetSS()
                    for s in range(0,samples):
                        npDataSS[b][s][f]=self.SS(npData[b][s][f])
            #return npDataSS
        elif len(npData.shape) == 2:
            samples, features = npData.shape
            npDataSS = np.zeros((samples, features), dtype=float)
            for f in range(0,features):
                self.resetSS()
                for s in range(0,samples):
                    npDataSS[s][f] = self.SS(npData[s][f])
            #return npDataSS
        elif type(npData) == list:
            npDataSS = []
            for s in range(0, len(npData)):
                npDataSS.append(self.SS(npData[s]))
            #return npDataSS
        else:
            print('Unkown numpy array diamensions: ' + str(npData.shape))

        #df_out = pd.DataFrame(npDataSS, columns=df.columns, dtype=float, index=df.index)  #
        #dfORIGSS.index.name = 'time'
        #df.values = npDataSS
        return npDataSS