import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from datetime import datetime

from A3C_DataCollector import Bitmex
from A3C_TRADING_ENV import Micro_Arch_4,Micro_Arch_Live_Test,Micro_Arch_Live
from DataNormalisation import DataNormalisation as NORM
import pandas as pd
import os
BMX = Bitmex()


class experience_buffer():
    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

def initFile(model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars / 2]):
        op_holder.append(tfVars[idx + total_vars / 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars / 2].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

def printActionsAnalyse(savedEntryActions,df):
        lag = 200 #number of rows
        plt.style.use('dark_background')
        plt.plot(df[['bidP']], c='b', label='bestBidPrice')

        INIT_AMOUNT = 10
        LEV = 1
        FINAL_AMOUNT = 0
        for row in savedEntryActions:

            tradeType = row[0]
            npData = row[1]
            x1_L = npData[9]
            x2_L = npData[10]#11
            y1_L = npData[5] #6
            y2_L = npData[7] #8

            x1_S = npData[9]
            x2_S = npData[11]#11
            y1_S = npData[6] #6
            y2_S = npData[8] #8
            #count,bidP,askP,pcentPriceChgLong,pcentPriceChgShort,entryPriceLong,entryPriceShort,exitPriceLong,exitPriceShort,entryIndex,exitIndexLong,exitIndexShort

            if tradeType == 'BUY':
                #(x1,x2)(y1,y2)
                pcentPriceChg = npData[3]  # 8
                plt.plot((int(x1_L), int(x2_L)), (y1_L, y2_L), c='g', label='BUY')
            elif tradeType == 'SELL':
                pcentPriceChg = npData[4]
                plt.plot((int(x1_S), int(x2_S)), (y1_S, y2_S), c='r', label='SELL')
            else:
                pcentPriceChg = 0

            INIT_AMOUNT=INIT_AMOUNT + INIT_AMOUNT*((pcentPriceChg/100)*LEV)
            print ('tradeType: ' + str(tradeType))

            print ('pcentPriceChg: ' + str(pcentPriceChg))
            print ('ROLL_AMOUNT: ' + str(INIT_AMOUNT))

        print('5 day span')
        plt.show()
        plt.hold(True)
        #exit()
        return

def printActionsAnalyseV2(savedEntryActions,df):
        plt.style.use('dark_background')
        plt.plot(df[['bidP']], c='b', label='bestBidPrice')

        INIT_AMOUNT = 10
        LEV = 1
        FINAL_AMOUNT = 0
        for row in savedEntryActions:
            # [tradeType[0],entryIndex[1], exitIndex[2], entryPrice[3], exitPrice[4],%chgPrice[5]]

            tradeType = row[0]
            #entryIndex = row[1]

            if tradeType == 'Long':
                pcentPriceChg = row[5]
                plt.plot((int(row[1]), int(row[2])), (row[3], row[4]), c='g', label='BUY')
            elif tradeType == 'Short':
                pcentPriceChg = row[5]
                plt.plot((int(row[1]), int(row[2])), (row[3], row[4]), c='r', label='SELL')
            else:
                pcentPriceChg = 0

            INIT_AMOUNT = INIT_AMOUNT + INIT_AMOUNT * ((pcentPriceChg / 100) * LEV)
            print ('tradeType: ' + str(tradeType))

            print ('pcentPriceChg: ' + str(pcentPriceChg))
            print ('ROLL_AMOUNT: ' + str(INIT_AMOUNT))

        print('5 day span')
        plt.show()
        plt.hold(True)
        #exit()
        return

def printActionsGraph(df):
    savedEntryActions = env.getEntryActionsLog()
    if isMicro_Arch_4 == False:
        printActionsAnalyseV2(savedEntryActions,df)
    else:
        printActionsAnalyseV2(savedEntryActions,df)

def calcRTAV2(INIT):
    savedEntryActions = env.getEntryActionsLog()
    INIT_AMOUNT =INIT
    for row in savedEntryActions:
        # [tradeType[0],entryIndex[1], exitIndex[2], entryPrice[3], exitPrice[4],%chgPrice[5]]

        tradeType = row[0]
        # entryIndex = row[1]

        if tradeType == 'Long':
            pcentPriceChg = row[5]
        elif tradeType == 'Short':
            pcentPriceChg = row[5]
        else:
            pcentPriceChg = 0

        INIT_AMOUNT = INIT_AMOUNT + (INIT_AMOUNT * ((pcentPriceChg / 100) * LEV))

    return INIT_AMOUNT
def calcRTA(INIT):
    savedEntryActions = env.getEntryActionsLog()

    INIT_AMOUNT =INIT
    for row in savedEntryActions:

        tradeType = row[0]
        npData = row[1]
        '''
        x1_L = npData[9]
        x2_L = npData[10]#11
        y1_L = npData[5] #6
        y2_L = npData[7] #8

        x1_S = npData[9]
        x2_S = npData[11]#11
        y1_S = npData[6] #6
        y2_S = npData[8] #8
        '''
        #count,bidP,askP,pcentPriceChgLong,pcentPriceChgShort,entryPriceLong,entryPriceShort,exitPriceLong,exitPriceShort,entryIndex,exitIndexLong,exitIndexShort

        if tradeType == 'BUY':
            #(x1,x2)(y1,y2)
            pcentPriceChg = npData[3]  # 8
        elif tradeType == 'SELL':
            pcentPriceChg = npData[4]
        else:
            pcentPriceChg = 0

        INIT_AMOUNT = INIT_AMOUNT + (INIT_AMOUNT*((pcentPriceChg/100)*LEV))

    return INIT_AMOUNT

def setupNormalisation(dir, lag, mode,tDelay, loadPrevDF):
    # DATA, DATA_ACTUAL_PRICE = BMX.loadBitmexMicroStructureDataV5(dir,1,lag,'testing',False)
    # DATA, DATA_ACTUAL_PRICE = BMX.collectDatav3(dir, lag, 'testing', loadPrevDF=False)
    DATA, DATA_ACTUAL_PRICE = BMX.collectDatav3(dir, lag, mode, tDelay,loadPrevDF=loadPrevDF)
    DATA_ACTUAL_PRICE_NO_VOL_THRESH = DATA_ACTUAL_PRICE
    '''
    plt.style.use('dark_background')
    plt.title(mode)
    plt.plot(DATA[['bidS_1']], c='b', label='BTSIZE')
    plt.plot(DATA[['askS_1']], c='r', label='STSIZE')

    #plt.plot(DATA_ACTUAL_PRICE[['bidP']],c='b',label='BID_PRICE')
    #plt.plot(DATA_ACTUAL_PRICE[['askP']],c='r',label='ASK_PRICE')

    # plt.axhline(0.3,c='y')
    # plt.axhline(-0.3,c='y')
    # plt.plot(NORM_DATA_1_CLIP[['VOI']],c='g',label='VOI')
    plt.show()
    plt.hold(True)
    '''
    print('BEFORE VOL THRESH')
    print('len(DATA): ' + str(len(DATA)))
    print('len(DATA_ACTUAL_PRICE): ' + str(len(DATA_ACTUAL_PRICE)))

    print('DATA: ' + str(DATA))
    print('DATA_ACTUAL_PRICE: ' + str(DATA_ACTUAL_PRICE))
    print('DATA.columns: ' + str(DATA.columns))
    newNpData = []
    newNpDataIndexes = []
    newNpDataActual = []
    newNpDataIndexesActual = []
    npData = DATA.as_matrix()
    npDataActual = DATA_ACTUAL_PRICE.as_matrix()
    indexVals = DATA.index.values
    indexValsActual = DATA_ACTUAL_PRICE.index.values
    for x in range(0, len(npData)):
        if npData[x][7] > VolThresh or npData[x][8] > VolThresh:
            newNpData.append(npData[x])
            newNpDataActual.append(npDataActual[x])
            newNpDataIndexes.append(indexVals[x])
            newNpDataIndexesActual.append(indexValsActual[x])
    npData = np.asarray(newNpData, dtype=float)
    df_Out = pd.DataFrame(data=npData, columns=DATA.columns, index=newNpDataIndexes, dtype=np.float32)
    df_OutActual = pd.DataFrame(data=newNpDataActual, columns=DATA_ACTUAL_PRICE.columns, index=newNpDataIndexesActual,
                                dtype=np.float32)

    DATA = df_Out
    DATA_ACTUAL_PRICE = df_OutActual

    print('AFTER VOL THRESH')
    print('len(DATA_ACTUAL_PRICE): ' + str(len(DATA_ACTUAL_PRICE)))
    print('DATA: ' + str(DATA))

    DATA = DATA.dropna()
    NORM_DATA_1 = NORM(MODE='STD').IN(DATA, isLog=False)
    NORM_ACTUAL_PRICE_1 = NORM(MODE='STD').IN(DATA_ACTUAL_PRICE, isLog=False)

    print('NORM_DATA_1.shape: ' + str(NORM_DATA_1.shape))
    print('NORM_ACTUAL_PRICE_1.shape: ' + str(NORM_ACTUAL_PRICE_1.shape))
    print('DATA_ACTUAL_PRICE: ' + str(DATA_ACTUAL_PRICE))
    return NORM_DATA_1,NORM_ACTUAL_PRICE_1,DATA_ACTUAL_PRICE,DATA_ACTUAL_PRICE_NO_VOL_THRESH

def LogSummary(episode_count,nbHolds,nbBuys,nbSells,epPercentagePriceChg,INIT,keep_prob,TSL_OFFSET):

    if isMicro_Arch_4 == False:
        RTA = calcRTAV2(INIT)
        #TSL_OFFSET = env.get_TSL_OFFSET()
    else:
        RTA = calcRTA(INIT)
    # Periodically save model parameters, and summary statistics.
    summary = tf.Summary()
    summary.value.add(tag='Perf/Total%PriceChg', simple_value=float(epPercentagePriceChg))
    summary.value.add(tag='Perf/RollingTradingAmount', simple_value=float(RTA))
    summary.value.add(tag='Perf/nbHolds', simple_value=float(nbHolds))
    summary.value.add(tag='Perf/nbBuys', simple_value=float(nbBuys))
    summary.value.add(tag='Perf/nbSells', simple_value=float(nbSells))
    summary.value.add(tag='Perf/keep_prob', simple_value=float(keep_prob))
    summary.value.add(tag='Perf/TSL_OFFSET', simple_value=float(TSL_OFFSET))

    summary_writer.add_summary(summary, episode_count)
    summary_writer.flush()


def splitdf(df):

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

class Q_Network():
    def __init__(self):
        nbActions = 3
        # These lines establish the feed-forward part of the network used to choose actions
        self.inputs = tf.placeholder(shape=[None,nbFeatures], dtype=tf.float32)
        self.Temp = tf.placeholder(shape=None, dtype=tf.float32)
        self.keep_per = tf.placeholder(shape=None, dtype=tf.float32)

        input = tf.contrib.layers.flatten(self.inputs)
        hidden = slim.fully_connected(input, 32, activation_fn=tf.nn.tanh, biases_initializer=None)
        hidden = slim.dropout(hidden, self.keep_per)
        hidden = slim.fully_connected(hidden, 128, activation_fn=tf.nn.tanh, biases_initializer=None)
        hidden = slim.dropout(hidden, self.keep_per)
        self.Q_out = slim.fully_connected(hidden, nbActions, activation_fn=None, biases_initializer=None)

        self.predict = tf.argmax(self.Q_out, 1)
        self.Q_dist = tf.nn.softmax(self.Q_out / self.Temp)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, nbActions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_onehot), axis=1)

        self.nextQ = tf.placeholder(shape=[None], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.nextQ - self.Q))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.updateModel = trainer.minimize(loss)


# Set learning parameters
exploration = "bayesian" #Exploration method. Choose between: greedy, random, e-greedy, boltzmann, bayesian.
y = .95 #Discount factor.
num_episodes = 145 #Total number of episodes to train network for.
tau = 0.001 #Amount to update target network at each step.
batch_size = 32 #Size of training batch
endE = 0.1 #Final chance of random action

anneling_steps = 130 #How many steps of training to reduce startE to endE.
pre_train_steps = 5000 #Number of steps used before training updates begin.
INIT_AMOUNT=10
LEV=1
STOP_LOSS_PERCENTAGE=50

model_path      = './model_meta_grid'
initFile(model_path)
save_model_chkpoint = 1
#startE = 1.0 #Starting chance of random action
startE = 0.1 #Starting chance of random action
models_to_keep = 5

VolThresh               = 1100000
lag = 300 #180 #3mins
TSL_OFFSET = 10
learning_rate = 0.00005
PERCENT_TO_TRADE = 20
mode = 'training' #'testing'
#mode = 'testing'
load_model = True#
save_model = False#True
train_model = False#True
loadPrevDF = True
tDelay = 1.0
hist_lookback= 3
nbFeatures = 9*hist_lookback
summary_writer = tf.summary.FileWriter(mode)

#env = gym.make('CartPole-v0')
#DATA_ACTUAL_PRICE_before_reduction=DATA_ACTUAL_PRICE
dir         = '/home/james/TRADING_PROJECT/DataCollector_BITMEX/BITMEX_DATA'
dirOanda    = '/home/james/TRADING_PROJECT/DataCollector_OANDA/OANDA_DATA'

NORM_DATA_1,NORM_ACTUAL_PRICE_1,DATA_ACTUAL_PRICE,DATA_ACTUAL_PRICE_NO_VOL_THRESH = setupNormalisation(dir, lag, mode,tDelay, loadPrevDF)
#env for perfect entry strat
#NORM_DATA_1,_=splitdf(NORM_DATA_1)
#NORM_ACTUAL_PRICE_1,_=splitdf(NORM_ACTUAL_PRICE_1)
#DATA_ACTUAL_PRICE,_=splitdf(DATA_ACTUAL_PRICE)
#DATA_ACTUAL_PRICE_NO_VOL_THRESH,_=splitdf(DATA_ACTUAL_PRICE_NO_VOL_THRESH)


isMicro_Arch_4 = True

env = Micro_Arch_4(
            INIT_AMOUNT,
            LEV,
            STOP_LOSS_PERCENTAGE,
            NORM_DATA_1,
            DATA_ACTUAL_PRICE,
            NORM_ACTUAL_PRICE_1,
    hist_lookback)

#testing/training for actual realistic trading
'''
isMicro_Arch_4 = False
env = Micro_Arch_Live_Test(INIT_AMOUNT,
                           tDelay,
                      LEV,
                      STOP_LOSS_PERCENTAGE,
                      TSL_OFFSET,
                           VolThresh,
                        dir,
                        mode,
                        loadPrevDF,hist_lookback,
                        USE_INIT_AMOUNT = False)

'''
#env = Micro_Arch_Live(INIT_AMOUNT,PERCENT_TO_TRADE, LEV, STOP_LOSS_PERCENTAGE, TSL_OFFSET,VolThresh,hist_lookback,USE_INIT_AMOUNT=False)

#INIT_AMOUNT,PERCENT_TO_TRADE, LEV, STOP_LOSS_PERCENTAGE, TSL_OFFSET,volThresh,dir,mode,loadPrevDF,USE_INIT_AMOUNT=False

tf.reset_default_graph()

q_net = Q_Network()
target_net = Q_Network()
#init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=models_to_keep)
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)
myBuffer = experience_buffer()

with tf.Session() as sess:
    #coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    #sess.run(init)
    updateTarget(targetOps, sess)
    e = startE
    stepDrop = (startE - endE) / anneling_steps
    total_steps = 0
    #init save for analysis
    #saver.save(sess, model_path + '/model-' + str(0) + '.cptk')
    #exit()
    #TSL = 18
    #nbToIgnore = 1
    for i in range(num_episodes):
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        start = datetime.utcnow()
        #env.setTSL(TSL)
        #env.setnbToIgnore(nbToIgnore)
        while d == False:
            j += 1

            # Choose an action using a sample from a dropout approximation of a bayesian q-network.
            a, allQ = sess.run([q_net.predict, q_net.Q_out],feed_dict={q_net.inputs: s, q_net.keep_per: (1 - e) + 0.1})
            a = a[0]
            s1, r, d, _ = env.step(a)
            #NP_ARRAY,INT,FLOAT,NP_ARRAY,BOOLEAN
            myBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))

            if total_steps > pre_train_steps and total_steps % 5 == 0 and train_model == True:
                # We use Double-DQN training algorithm
                trainBatch = myBuffer.sample(batch_size)
                Q1 = sess.run(q_net.predict, feed_dict={q_net.inputs: np.vstack(trainBatch[:, 3]), q_net.keep_per: 1.0})
                Q2 = sess.run(target_net.Q_out,feed_dict={target_net.inputs: np.vstack(trainBatch[:, 3]), target_net.keep_per: 1.0})
                end_multiplier = -(trainBatch[:, 4] - 1)
                doubleQ = Q2[range(batch_size), Q1]
                targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                _ = sess.run(q_net.updateModel,feed_dict={q_net.inputs: np.vstack(trainBatch[:, 0]), q_net.nextQ: targetQ,q_net.keep_per: 1.0, q_net.actions: trainBatch[:, 1]})
                updateTarget(targetOps, sess)

            rAll += r
            s = s1
            total_steps += 1
            if d == True:
                #print ('break')
                break


        if e > endE and total_steps > pre_train_steps:
            e -= stepDrop


        if i % save_model_chkpoint == 0 and save_model == True:
            saver.save(sess, model_path + '/model-' + str(i) + '.cptk')
            print("Saved Model")

        [epPercentagePriceChg, nbBuy, nbHold, nbSell] = env.prntPerformance()
        print " epPercentagePriceChg: " + str(epPercentagePriceChg) + " keep prob: " + str((1 - e) + 0.1)
        LogSummary(i,nbBuy, nbHold, nbSell,epPercentagePriceChg,INIT_AMOUNT,(1 - e) + 0.1,TSL_OFFSET)
        #TSL = TSL + 1
        #nbToIgnore = nbToIgnore + 1
printActionsGraph(DATA_ACTUAL_PRICE_NO_VOL_THRESH)
