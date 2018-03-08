import numpy as np
import pandas as pd

class DataNormalisation:

    maxMinListForColumns = 0
    col_input_dims = 0
    normColList = []
    '''
    #Winsorise data and limit the outliers that are above 95 percentile and below 5 percentile
        a = np.array([92, 19, 101, 58, 1053, 91, 26, 78, 10, 13, -40, 101, 86, 85, 15, 89, 89, 28, -5, 41])
        scipy.stats.mstats.winsorize(a, limits=0.05)
    '''
    ###NORMALISATION METHODS###
    ##API##
    def StandardizationIN(self,npData):
        #Subtract the mean from each Feature(column)
        #Divide each Feature by its Standard Deviation
        cols = []
        for col in npData.T:
            mean    = np.mean(col)
            SD      = np.std(col)
            StdCol=(col-mean)/SD
            cols.append(StdCol)
            print ('mean    : ' + str(mean))
            print ('SD      : ' + str(SD))
            print ('col     : ' + str(col))
            print ('StdCol  : ' + str(StdCol))

        stdnpData = np.asarray(cols,float).T
        print ('stdnpData: ' + str(stdnpData))
        return stdnpData

    ###API#######
    def IN(self,df,windowSize = 10,zscoreLim=1,isLog = False,verbose=True,updateMinMax= False):
        df=df.astype('float32')
        if updateMinMax == True:
            if self.FirstRun == True:
                print('Setting up MinMax.')
                self.setMinMax(df, windowSize, isLog)
            else:
                print('Updating MinMax.')
                self.updateMinMax(df, windowSize, isLog)
        else:
            if self.FirstRun == True:
                print('Setting up MinMax.')
                self.setMinMax(df, windowSize, isLog)
            else:
                pass
                #print('MinMax values are setup.')
        if self.MODE == 'STD':
            return self.IN_Standard(df,isLog)
        elif self.MODE == 'STD_SD_CLIP':
            return self.IN_STD_Clip(df,windowSize,zscoreLim,isLog)
        elif self.MODE == 'STDDEV':
            return self.IN_Std_Dev(df, windowSize, isLog)
        elif self.MODE == 'ZSCORE':
            return self.IN_Z_Scores(df, windowSize)
        elif self.MODE == 'DIFF':
            return self.IN_Diff(df, verbose, isLog)
        else:
            print('NORMALISATION MODE ERROR!')
            exit()
    def OUT(self,df,isLog = False):
        df = df.astype('float32')
        if self.MODE == 'STD':
            return self.OUT_Standard(df)
        elif self.MODE == 'STDDEV':
            return self.OUT_Std_Dev(df, isLog)
        elif self.MODE == 'ZSCORE':
            return self.OUT_Z_Scores(df)
        elif self.MODE == 'DIFF':
            return self.OUT_Diff(df, isLog)
        else:
            print('NORMALISATION MODE ERROR!')
            exit()
    def setMinMax(self,df,windowSize,isLog):

        if self.MODE == 'STD':
            self.setMinMaxForCols_Standard(df,isLog)
        elif self.MODE == 'STD_SD_CLIP':
            self.setMinMaxForCols_Standard(df, isLog)
        elif self.MODE == 'STDDEV':
            self.setMinMaxForCols_Std_Dev(df, windowSize,isLog)
        elif self.MODE == 'ZSCORE':
            self.setMinMaxForCols_Z_Scores(df, windowSize)
        elif self.MODE == 'DIFF':
            self.setMinMaxForCols_Diff(df,isLog )
        else:
            print('NORMALISATION MODE ERROR!')
            exit()
        self.FirstRun = False
        return
    def updateMinMax(self,df,windowSize,isLog):

        if self.MODE == 'STD':
            self.updateMinMaxForCols_Standard(df,isLog)
        elif self.MODE == 'STD_SD_CLIP':
            self.updateMinMaxForCols_Standard(df, isLog)
        elif self.MODE == 'STDDEV':
            self.updateMinMaxForCols_Std_Dev(df, windowSize,isLog)
        elif self.MODE == 'ZSCORE':
            self.updateMinMaxForCols_Z_Scores(df, windowSize)
        elif self.MODE == 'DIFF':
            self.updateMinMaxForCols_Diff(df, isLog)
        else:
            print('NORMALISATION MODE ERROR!')
            exit()
        return
    ####NORM IN METHODS############

    ##############
    # Version 1
    def IN_Standard(self,df,isLog=False):
        if isLog == True:
            df_log = np.log(df + 1e-7)
        else:
            df_log = df
        npDataCpy = df_log.as_matrix()

        #Global min and max of given time series of prices
        self.LocalMinMaxList = []
        npDataTrans = npDataCpy.T
        for n in range(0,len(npDataTrans)):
            max = np.max(npDataTrans[n])
            min = np.min(npDataTrans[n])
            self.LocalMinMaxList.append((n, min, max))

            if self.MinMaxList[n][1] > min:
                print('New min found. MinMax broke.')
                print('Current: ' + str(self.MinMaxList[n][1]))
                print('found: ' + str(min))
            if self.MinMaxList[n][2] < max:
                print('New max found. MinMax broke.')
                print('Current: ' + str(self.MinMaxList[n][2]))
                print('found: ' + str(max))


            #self.MinMaxList.append((n, min, max))
                #print('npDataCpy: ' + str(npDataCpy))
        npDataGradNorm = self.formatDataForModelIN(npDataCpy,self.MinMaxList)

        df_Out = pd.DataFrame(data=npDataGradNorm, columns=df.columns, index=df.index,dtype=np.float32)
        return df_Out #[b][t][f]
    # Version 2
    def IN_Std_Dev(self, df,windowSize, isLog = False):
        #print('Calculates rolstd')
        #print('df: ' +str(df))
        #exit()
        rolstd = df.rolling(window=windowSize+1,min_periods=1, center=False).std()  # pd.rolling_std(timeseries, window=12)
        if isLog == True:
            ts_log_std = np.log(rolstd + 1e-7)  # all values need to be positive
        else:
            ts_log_std = rolstd


        df1 = pd.concat([ts_log_std], axis=1)  # .dropna(inplace=True)
        df1 = df1.dropna()
        npData = df1.as_matrix().astype(float)

        # Global min and max of given time series of prices
        self.LocalMinMaxListv2 = []
        npDataTrans = npData.T
        for n in range(0, len(npDataTrans)):
            max = np.max(npDataTrans[n])
            min = np.min(npDataTrans[n])
            self.LocalMinMaxListv2.append((n, min, max))
            if self.MinMaxListv2[n][1] > min:
                print('New min found. MinMax broke.')
                print('Current: ' + str(self.MinMaxListv2[n][1]))
                print('found: ' + str(min))
            if self.MinMaxListv2[n][2] < max:
                print('New max found. MinMax broke.')
                print('Current: ' + str(self.MinMaxListv2[n][2]))
                print('found: ' + str(max))

                # self.MinMaxList.append((n, min, max))

        npDataNorm = self.formatDataForModelIN(npData, self.MinMaxListv2)
        # print('npDataDiffNorm: ' + str(npDataDiffNorm))
        # add rows of nans on top of np array
        rows, columns = npDataNorm.shape
        #print(rows, columns)

        topRow = [np.nan for _ in range(int(columns))]
        npDataNorm = np.vstack((topRow, npDataNorm))
        rows, columns = npDataNorm.shape
        #print(rows, columns)

        df_Norm = pd.DataFrame(data=npDataNorm, columns=df.columns, index=df.index,dtype=np.float32).bfill()
        return df_Norm
    # Version 3
    def IN_Z_Scores(self, df,windowSize):
        df = df.dropna()
        ts_log = np.log(df + 1e-7)             #all values need to be positive
        print('Calculates rolmean, rolstd, ts_z_score')
        #ts_log_diff = ts_log - ts_log.shift()
        #ts_log = np.log(ts)
        rolmean = ts_log.rolling(window=windowSize, center=False).mean().bfill()  # pd.rolling_mean(timeseries, window=12)
        rolstd = ts_log.rolling(window=windowSize, center=False).std().bfill()  # pd.rolling_std(timeseries, window=12)
        ts_z_score = (ts_log - rolmean) / rolstd
        #ts_z_score
        df1 = pd.concat([rolmean, rolstd, ts_z_score], axis=1) #.dropna(inplace=True)
        print('npData.shape: ' + str(df1.as_matrix().shape))
        df1 = df1.dropna()
        npData = df1.as_matrix().astype(float)
        #print('npData: ' + str(npData))
        print('npData.shape: ' + str(npData.shape))
        cols = df.columns
        print('df.shape: ' + str(df.shape))
        new_cols = []
        tags = []
        tags.append('_MEAN')
        tags.append('_STDDEV')
        tags.append('_ZSCORE')
        for t in tags:
            for c in cols:
                new_cols.append(str(c)+t)
        print('new_cols: ' +str(new_cols))

        # Global min and max of given time series of prices
        self.LocalMinMaxListv3 = []
        npDataTrans = npData.T
        for n in range(0, len(npDataTrans)):
            max = np.max(npDataTrans[n])
            min = np.min(npDataTrans[n])
            self.LocalMinMaxListv3.append((n, min, max))
            if self.MinMaxListv3[n][1] > min:
                print('New min found. MinMax broke.')
                print('Current: ' + str(self.MinMaxListv3[n][1]))
                print('found: ' + str(min))
            if self.MinMaxListv3[n][2] < max:
                print('New max found. MinMax broke.')
                print('Current: ' + str(self.MinMaxListv3[n][2]))
                print('found: ' + str(max))

                # self.MinMaxList.append((n, min, max))

        npDataNorm = self.formatDataForModelIN(npData, self.MinMaxListv3)
        print('len(npDataNorm): ' + str(len(npDataNorm)))
        print('len(df.index)  : ' + str(len(df.index)))

        df_Norm = pd.DataFrame(data=npDataNorm, columns=new_cols, index=df1.index, dtype=np.float32).bfill()
        return df_Norm

    def IN_STD_Clip(self, df, windowSize,zscoreLim,isLog = False):
        if isLog == True:
            df_log = np.log(df + 1e-7)
        else:
            df_log = df
        print('Calculates rolmean, rolstd, ts_z_score')
        # ts_log_diff = ts_log - ts_log.shift()
        # ts_log = np.log(ts)
        rolmean = df_log.rolling(window=windowSize,
                                 center=False).mean().bfill()  # pd.rolling_mean(timeseries, window=12)
        rolstd = df_log.rolling(window=windowSize, center=False).std().bfill()  # pd.rolling_std(timeseries, window=12)
        ts_z_score = (df_log - rolmean) / rolstd
        npBuff = ts_z_score.as_matrix()
        buff = np.zeros((npBuff.shape),dtype=float)
        i = 0
        for x in npBuff:
            print x
            if x >= 0:
                pass
            else:
                x = -1*x
            if x > zscoreLim:
                x = zscoreLim
            buff[i] = x
            i = i+1

        npDataCpy = df_log.as_matrix()

        # Global min and max of given time series of prices
        self.LocalMinMaxList = []
        npDataTrans = npDataCpy.T
        for n in range(0, len(npDataTrans)):
            max = np.max(npDataTrans[n])
            min = np.min(npDataTrans[n])
            self.LocalMinMaxList.append((n, min, max))

            if self.MinMaxList[n][1] > min:
                print('New min found. MinMax broke.')
                print('Current: ' + str(self.MinMaxList[n][1]))
                print('found: ' + str(min))
            if self.MinMaxList[n][2] < max:
                print('New max found. MinMax broke.')
                print('Current: ' + str(self.MinMaxList[n][2]))
                print('found: ' + str(max))


                # self.MinMaxList.append((n, min, max))
                # print('npDataCpy: ' + str(npDataCpy))
        npDataGradNorm = self.formatDataForModelIN(npDataCpy, self.MinMaxList)

        df_Out = pd.DataFrame(data=npDataGradNorm, columns=df.columns, index=df.index, dtype=np.float32)
        return df_Out  # [b][t][f]

    # Version 4
    def IN_Diff(self, df,verbose = True,isLog = False):
        self.df_orig = df.copy()
        if isLog == True:
            df_log = np.log(df + 1e-7)
        else:
            df_log = df
        npData = (df_log - df_log.shift()).dropna().as_matrix()
        self.LocalMinMaxListv4 = []
        npDataTrans = npData.T
        for n in range(0, len(npDataTrans)):
            max = np.max(npDataTrans[n])
            min = np.min(npDataTrans[n])
            self.LocalMinMaxListv4.append((n, min, max))
            if self.MinMaxListv4[n][1] > min:
                print('New min found. MinMax broke.')
                print('Current: ' + str(self.MinMaxListv4[n][1]))
                print('found: ' + str(min))
            if self.MinMaxListv4[n][2] < max:
                print('New max found. MinMax broke.')
                print('Current: ' + str(self.MinMaxListv4[n][2]))
                print('found: ' + str(max))
        npDataDiffNorm = self.formatDataForModelIN(npData, self.MinMaxListv4)
        #add rows of nans on top of np array
        rows,columns = npDataDiffNorm.shape
        #print(rows,columns)

        topRow = [np.nan for _ in range(int(columns))]
        npDataDiffNorm = np.vstack((topRow,npDataDiffNorm))
        rows, columns = npDataDiffNorm.shape
        #print(rows, columns)

        df_Norm = pd.DataFrame(data=npDataDiffNorm, columns=df.columns, index=df.index[:],dtype=np.float32).bfill()
        #print('df_Norm: ' + str(df_Norm))
        if verbose == True:
            self.getLocalSearchSpacePercentUsed()
        #print('***Diff norm end***')
        #firstDate = str(df_Norm.head(1).index.tolist()[0].to_pydatetime())
        #latestDate = str(df_Norm.tail(1).index.tolist()[0].to_pydatetime())
        #print('firstDate : ' + str(firstDate))
        #print('latestDate: ' + str(latestDate))
        self.df_diff_orig =df_Norm
        return df_Norm
    #################################

    ####NORM OUT METHODS############
    # Version 1
    def OUT_Standard(self, df,isLog=False):
        npDataCpy = df.as_matrix()
        npDataIN = self.formatDataForModelOUT(npDataCpy,self.getMinMaxForCols())
        if isLog == True:
            npDataOut = np.exp(npDataIN)
        else:
            npDataOut = npDataIN

        df_Out = pd.DataFrame(data=npDataOut, columns=df.columns, dtype=np.float32)
        return df_Out
    # Version 2
    def OUT_Std_Dev(self, df, isLog = False):
        npData = df.as_matrix()
        npDataCpy = np.copy(npData)
        np_log = self.formatDataForModelOUT(npDataCpy, self.getMinMaxForColsv2())
        if isLog == True:
            npDataOut = np.exp(np_log)
        else:
            npDataOut = np_log
        df_Out = pd.DataFrame(data=npDataOut, columns=df.columns,dtype=np.float32)
        # print('npDataDiffNorm: ' + str(npDataDiffNorm))
        return df_Out
    # Version 3
    def OUT_Z_Scores(self, df):
        npDataCpy = df.as_matrix()
        npDataIN = self.formatDataForModelOUT(npDataCpy,self.getMinMaxForColsv3())
        examples,features = npDataIN.shape
        np_log = np.zeros((examples,features),dtype=np.float32)
        #[rolmean, rolstd, ts_z_score] (200,3)
        for i in range(0,examples):
            np_log[i] = (npDataIN[i][2]*npDataIN[i][1])+npDataIN[i][0]
        #np_log = (z_score*rolstd)+rolmean
        npDataOut = np.exp(np_log)
        df_Out = pd.DataFrame(data=npDataOut, columns=df.columns, index=df.index,dtype=np.float32)
        return df_Out
    # Version 4
    def OUT_Diff(self, dfNormDiff,isLog = False):
        #dfOrig=(self.df_diff_orig - self.df_diff_orig.shift()).dropna()
        #print('dfOrig: ' + str(self.df_diff_orig[:5]))
        #print('dfNormDiff[:5]: ' + str(dfNormDiff[:5]))
        #isdfValsEqual = afe(self.df_diff_orig[:5].as_matrix(),dfNormDiff[:5])
        isdfValsEqual = np.array_equal(self.df_diff_orig[:5].as_matrix(),dfNormDiff[:5].as_matrix())

        if isdfValsEqual == False:
            print('dataframes not equal')
            print('self.df_diff_orig[:5]: ' +str(self.df_diff_orig[:5]))
            print('dfNormDiff[:5]: ' + str(dfNormDiff[:5]))
            return pd.DataFrame()
        npData = dfNormDiff.as_matrix()
        npDataIN = self.df_orig.as_matrix()

        npDataCpy = np.copy(npData)
        OrigUnDiffVal = []
        examples, features = npDataIN.shape
        for f in range(0, features):
            OrigUnDiffVal.append(npDataIN[0][f])  # -1 might need to change
        npDataDiffLog = self.formatDataForModelOUT(npDataCpy,self.getMinMaxForColsv4())
        examples, features = npDataDiffLog.shape
        unDiffLog = np.zeros((examples+1, features),dtype=float) #+1 for known original value

        #init original values
        for f in range(0, features):
            if isLog == True:
                unDiffLog[0][f] = np.log(OrigUnDiffVal[f])
            else:
                unDiffLog[0][f] = OrigUnDiffVal[f]

        for e in range(0,examples):
            for f in range(0, features):
                unDiffLog[e+1][f] = unDiffLog[e][f] + npDataDiffLog[e][f]
        if isLog == True:
            npDataOut = np.exp(unDiffLog)
        else:
            npDataOut = unDiffLog

        origFirstDate = str(self.df_diff_orig.head(1).index.tolist()[0])
        periods = len(dfNormDiff)+1
        #rng = pd.date_range(start=origFirstDate, periods=periods, freq=str(self.tDeltaInMins)+'T')

        df_Out = pd.DataFrame(data=npDataOut, columns=self.df_diff_orig.columns ,dtype=np.float32)
        return df_Out
    #################################

    def formatDataForModelIN(self, datanp,MinMaxList):
        # print('formatDataForModelIN')
        datanpCpy = np.copy(datanp)

        # normalise each COLUMN
        for t in range(datanpCpy.shape[0]):
            for f in range(datanpCpy.shape[1]):
                #print('datanpCpy[t][f]: ' + str(datanpCpy[t][f]))

                datanpCpy[t][f] = self.normForwardSingle(datanpCpy[t][f], MinMaxList[f][2], MinMaxList[f][1])  # MinMax[min=0,max=1][timestamp=0,colslist[1]=1]

        return datanpCpy
    def formatDataForModelOUT(self, datanp,MinMaxList):
        #print('formatDataForModelOUT')
        datanpCpy = np.copy(datanp)

        for t in range(datanpCpy.shape[0]):
            for f in range(datanpCpy.shape[1]):
                datanpCpy[t][f] = self.normReverseSingle(datanpCpy[t][f], MinMaxList[f][2], MinMaxList[f][1])  # MinMax[min=0,max=1][timestamp=0,colslist[1]=1]

        return datanpCpy
    def normForwardSingle(self, D, Dmax, Dmin):


        if self.isTanhOn == True:
            if self.featRangeMode == 1:
                rescaled = ((float(D) - ((float(Dmax) + float(Dmin)) / float(2))) / ((float(Dmax) - float(Dmin)) / float(2))) * float(self.tanhConstant)  # (+2,-2)
            else:
                print('tanh and a scale of 0 to 1 does not work')
                exit()
                rescaled =((float(D) - float(Dmin))/(float(Dmax) - float(Dmin))) * float(self.tanhConstant)

            afterTanh = np.tanh(float(rescaled))#tanh or custom SIGMOID between +1, -1
            #out = x/(sqrt(1+x**2)) #less steep than tanh
            return afterTanh
        else:
            if self.featRangeMode == 1:
                rescaled = ((float(D) - ((float(Dmax) + float(Dmin)) / float(2))) / ((float(Dmax) - float(Dmin)) / float(2)))  # (+2,-2)
            else:
                rescaled = ((float(D) - float(Dmin)) / (float(Dmax) - float(Dmin)))

            return rescaled
    def normReverseSingle(self, afterTanh, Dmax, Dmin):
        if self.isTanhOn == True:
            beforeTanh = np.arctanh(float(afterTanh))  # REVERSE Tanh
            #out = x/(sqrt(1+x**2)) #less steep than tanh
            if self.featRangeMode == 1:
                unscaled = (float(beforeTanh) / float(self.tanhConstant)) * ((float(Dmax) - float(Dmin)) / float(2)) + ((float(Dmax) + float(Dmin)) / float(2))  # (+2,-2)
            else:
                print('tanh and a scale of 0 to 1 does not work')
                exit()
                unscaled = (((float(beforeTanh) / float(self.tanhConstant)))*(float(Dmax)-float(Dmin)))+float(Dmin)
            return unscaled
        else:
            if self.featRangeMode == 1:
                unscaled = (float(afterTanh)) * ((float(Dmax) - float(Dmin)) / float(2)) + ((float(Dmax) + float(Dmin)) / float(2))  # (+2,-2)
            else:
                unscaled = ((float(afterTanh)) * (float(Dmax) - float(Dmin))) + float(Dmin)
            return unscaled



        return unscaled
    #######
    def setMinMaxForCols_Standard(self, df,isLog = False):
        #find global min and max price values within historical data

        #Transpose and loop through columns
        #print ('setMinMaxForCols_Standard')
        if isLog == True:
            df_log = np.log(df + 1e-7)
        else:
            df_log = df

        npData = df_log.as_matrix()
        for n in range(0,len(npData.T)):
            #print ('len(npData.T): ' + str(len(npData.T)))
            #print ('npData.T: ' + str(npData.T))
            #print ('npData.T[n]: ' + str(npData.T[n]))
            self.MinMaxList.append((n,min(npData.T[n]),max(npData.T[n])))
        # (X_ROWS,3_COLUMNS)
        # ((FEATURE,MIN,MAX),(FEATURE,MIN,MAX))
        #self.MinMaxList = []
        return self.MinMaxList
    def setMinMaxForCols_Std_Dev(self, df,windowSize, isLog = False):
        #print ('setMinMaxForCols_Std_Dev')
        #print ('df.head(): ' + str(df.head()))
        rolstd = df.rolling(window=windowSize+1, center=False).std()  # pd.rolling_std(timeseries, window=12)
        #print ('rolstd.head(20): ' + str(rolstd.head(20)))
        if isLog == True:
            ts_log_std = np.log(rolstd + 1e-7)  # all values need to be positive
        else:
            ts_log_std = rolstd


        df1 = pd.concat([ts_log_std], axis=1)  # .dropna(inplace=True)
        #print ('df1.head(): ' + str(df1.head()))
        df1 = df1.dropna()
        npData = df1.as_matrix().astype(float)
        #print ('npData: ' + str(npData))
        # Transpose and loop through columns
        npDataTrans = npData.T
        for n in range(0, len(npDataTrans)):
            #print ('len(npData.T): ' + str(len(npData.T)))
            #print ('npData.T: ' + str(npData.T))
            #print ('npData.T[n]: ' + str(npData.T[n]))
            self.MinMaxListv2.append((n, min(npData.T[n]), max(npData.T[n])))

        return self.MinMaxListv2
    def setMinMaxForCols_Z_Scores(self, df,windowSize):
        ts_log = np.log(df + 1e-7)  # all values need to be positive
        # print('ts_log: ' + str(ts_log))
        # ts_log_diff = ts_log - ts_log.shift()
        # ts_log = np.log(ts)

        rolmean = ts_log.rolling(window=windowSize, center=False).mean()  # pd.rolling_mean(timeseries, window=12)
        rolstd = ts_log.rolling(window=windowSize, center=False).std()  # pd.rolling_std(timeseries, window=12)
        ts_z_score = (ts_log - rolmean) / rolstd

        # print(len(ts_log))
        # print(len(rolmean))
        # print(len(rolstd))
        # print(len(ts_z_score))
        df1 = pd.concat([rolmean, rolstd, ts_z_score], axis=1)  # .dropna(inplace=True)
        df1 = df1.dropna()
        npData = df1.as_matrix().astype(float)
        # Transpose and loop through columns
        npDataTrans = npData.T
        for n in range(0, len(npDataTrans)):
            self.MinMaxListv3.append((n, min(npData.T[n]), max(npData.T[n])))
            #Differentitation
            #colsDiff = []
            #for col in range(0, len(npDataTrans[n]) - 1):
            #    colsDiff.append(npDataTrans[n][col] - npDataTrans[n][col + 1])
            #maxGrad = max(colsDiff)
            #minGrad = min(colsDiff)
            #self.MinMaxListv3.append((n, minGrad, maxGrad))

        # (X_ROWS,3_COLUMNS)
        # ((FEATURE,MIN,MAX),(FEATURE,MIN,MAX))
        # self.MinMaxList = []
        return self.MinMaxListv3
    def setMinMaxForCols_Diff(self, df,isLog = False):
        #set global min max

        if isLog == True:
            df_log = np.log(df + 1e-7)
        else:
            df_log = df

        npData = (df_log - df_log.shift()).dropna().as_matrix()
        self.MinMaxListv4 = []
        npDataTrans = npData.T
        for n in range(0, len(npDataTrans)):
            max = np.max(npDataTrans[n])
            min = np.min(npDataTrans[n])
            self.MinMaxListv4.append((n, min, max))

    def updateMinMaxForCols_Standard(self, df,isLog =False):
        #MinMax to update
        if isLog == True:
            df_log = np.log(df + 1e-7)
        else:
            df_log = df
        self.MinMaxList = self.setNewMinMax(df_log, self.MinMaxList)

        return self.MinMaxList
    def updateMinMaxForCols_Std_Dev(self, df,windowSize, isLog = False):
        rolstd = df.rolling(window=windowSize, center=False).std()  # pd.rolling_std(timeseries, window=12)
        if isLog == True:
            ts_log_std = np.log(rolstd + 1e-7)  # all values need to be positive
        else:
            ts_log_std = rolstd


        df1 = pd.concat([ts_log_std], axis=1)  # .dropna(inplace=True)
        df1 = df1.dropna()
        # MinMax to update
        self.MinMaxListv2 = self.setNewMinMax(df1, self.MinMaxListv2)

        return self.MinMaxListv2
    def updateMinMaxForCols_Z_Scores(self, df,windowSize):
        ts_log = np.log(df + 1e-7)  # all values need to be positive
        # print('ts_log: ' + str(ts_log))
        # ts_log_diff = ts_log - ts_log.shift()
        # ts_log = np.log(ts)

        rolmean = ts_log.rolling(window=windowSize, center=False).mean()  # pd.rolling_mean(timeseries, window=12)
        rolstd = ts_log.rolling(window=windowSize, center=False).std()  # pd.rolling_std(timeseries, window=12)
        ts_z_score = (ts_log - rolmean) / rolstd

        # print(len(ts_log))
        # print(len(rolmean))
        # print(len(rolstd))
        # print(len(ts_z_score))
        df1 = pd.concat([rolmean, rolstd, ts_z_score], axis=1)  # .dropna(inplace=True)
        df1 = df1.dropna()
        # MinMax to update
        self.MinMaxListv3 = self.setNewMinMax(df1, self.MinMaxListv3)
        return self.MinMaxListv3
    def updateMinMaxForCols_Diff(self, df,isLog = False):
        #set global min max

        if isLog == True:
            df_log = np.log(df + 1e-7)
        else:
            df_log = df

        df1 = (df_log - df_log.shift()).dropna()
        self.MinMaxListv4 = self.setNewMinMax(df1, self.MinMaxListv4)
        return self.MinMaxListv4

    def setNewMinMax(self, df,MinMax):
        # set global min max

        npData = df.as_matrix()
        npDataTrans = npData.T
        for n in range(0, len(npDataTrans)):
            max = np.max(npDataTrans[n])
            min = np.min(npDataTrans[n])
            if MinMax[n][1] > min:
                #new min
                prev_n, prev_min, prev_max = MinMax[n]
                MinMax[n] = (prev_n, min, prev_max)
            if MinMax[n][2] < max:
                #new max
                prev_n, prev_min, prev_max = MinMax[n]
                MinMax[n] = (prev_n, prev_min, max)

        return MinMax

    def getLocalMinMaxForCols(self):
        # (X_ROWS,3_COLUMNS)
        # ((FEATURE,MIN,MAX),(FEATURE,MIN,MAX))
        return self.LocalMinMaxList
        ###########################
    def getLocalMinMaxForColsv2(self):
        # (X_ROWS,3_COLUMNS)
        # ((FEATURE,MIN,MAX),(FEATURE,MIN,MAX))
        return self.LocalMinMaxListv2
        ###########################
    def getLocalMinMaxForColsv3(self):
        # (X_ROWS,3_COLUMNS)
        # ((FEATURE,MIN,MAX),(FEATURE,MIN,MAX))
        return self.LocalMinMaxListv3
        ###########################
    def getLocalMinMaxForColsv4(self):
        # (X_ROWS,3_COLUMNS)
        # ((FEATURE,MIN,MAX),(FEATURE,MIN,MAX))
        return self.LocalMinMaxListv4
        ###########################

    def getMinMaxForCols(self):
        # (X_ROWS,3_COLUMNS)
        # ((FEATURE,MIN,MAX),(FEATURE,MIN,MAX))
        return self.MinMaxList
        ###########################

    def getMinMaxForColsv2(self):
        # (X_ROWS,3_COLUMNS)
        # ((FEATURE,MIN,MAX),(FEATURE,MIN,MAX))
        return self.MinMaxListv2
        ###########################
    def getMinMaxForColsv3(self):
        # (X_ROWS,3_COLUMNS)
        # ((FEATURE,MIN,MAX),(FEATURE,MIN,MAX))
        return self.MinMaxListv3
        ###########################
    def getMinMaxForColsv4(self):
        # (X_ROWS,3_COLUMNS)
        # ((FEATURE,MIN,MAX),(FEATURE,MIN,MAX))
        return self.MinMaxListv4
        ###########################

    def setTanhConstant(self,tanhConstant):
        print('min is 1   giving 0.76159415595. (1 - 0.76159415595) = room to move if limits break.')
        print('max is 2.8 giving 0.99263152020. (1 - 0.99263152020) = room to move if limits break.')
        print('limits break if future time series data has min and max values bigger than the ones set beforehand.')

        if(tanhConstant<1):
            print ('Constant too low.')
        elif(tanhConstant>2.8):
            print ('Constant too high.')
        else:
            print ('Constant is just right.')
            self.tanhConstant = tanhConstant
        return self.tanhConstant
    def getTanhConstant(self):
        return self.tanhConstant

    def getLocalSearchSpacePercentUsed(self):
        minMax          = self.getMinMaxForCols()
        localMinMax     = self.getLocalMinMaxForCols()
        minMaxv2        = self.getMinMaxForColsv2()
        localMinMaxv2   = self.getLocalMinMaxForColsv2()
        minMaxv3        = self.getMinMaxForColsv3()
        localMinMaxv3   = self.getLocalMinMaxForColsv3()
        minMaxv4        = self.getMinMaxForColsv4()
        localMinMaxv4   = self.getLocalMinMaxForColsv4()

        if minMax != [] and localMinMax != []:
            #V1 used
            globalDiff = minMax[0][2] - minMax[0][1]
            localDiff  = localMinMax[0][2]-localMinMax[0][1]
            searchSpacePercentage = (localDiff/globalDiff)*float(100)
            print('STD    Mode, input array using ' + str(searchSpacePercentage) + '% of Min Max range.')
            #return
        if minMaxv2 != [] and localMinMaxv2 != []:
            #V2 used
            globalDiffv2 = minMaxv2[0][2]       - minMaxv2[0][1]
            localDiffv2 = localMinMaxv2[0][2]   - localMinMaxv2[0][1]
            searchSpacePercentage = (localDiffv2 / globalDiffv2) * float(100)
            print('STDDEV Mode, input array using ' + str(searchSpacePercentage) + '% of Min Max range.')
            #return
        if minMaxv3 != [] and localMinMaxv3 != []:
            # V2 used
            globalDiffv2 = minMaxv3[0][2] - minMaxv3[0][1]
            localDiffv2 = localMinMaxv3[0][2] - localMinMaxv3[0][1]
            searchSpacePercentage = (localDiffv2 / globalDiffv2) * float(100)
            print('ZSCORE Mode, input array using ' + str(searchSpacePercentage) + '% of Min Max range.')
            # return
        if minMaxv4 != [] and localMinMaxv4 != []:
            # V2 used
            globalDiffv4 = minMaxv4[0][2] - minMaxv4[0][1]
            localDiffv4 = localMinMaxv4[0][2] - localMinMaxv4[0][1]
            searchSpacePercentage = (localDiffv4 / globalDiffv4) * float(100)
            print('DIFF   Mode, input array using ' + str(searchSpacePercentage) + '% of Min Max range.')
            # return
        return

    def checkMinMaxLimit(self):
        minMax = self.getMinMaxForCols()
        minMaxv2 = self.getMinMaxForColsv2()
        localMinMax = self.getLocalMinMaxForCols()
        localMinMaxv2 = self.getLocalMinMaxForColsv2()

        #loop through all columns
        if minMax != [] and localMinMax != []:
            for nCol in range(0,len(minMax)):
                # check minimum lower bound
                if localMinMax[nCol][1]<minMax[nCol][1]:
                    outsideMinPercent = ((minMax[nCol][1]-localMinMax[nCol][2]) / (minMax[nCol][2] - minMax[nCol][1])) * 100
                    print('Minimum minMax limit operating ' + str(outsideMinPercent) + ' percent below global set minimum.')
                    # absolute min max boundary is -2.8 +2.8
                    rescaled = ((localMinMax[nCol][1] - ((minMax[nCol][2] + minMax[nCol][1]) / 2)) / ((minMax[nCol][2] - minMax[nCol][1]) / 2)) * self.tanhConstant
                    if rescaled >= -2.8:
                        print('absolute min max boundary is -2.8 +2.8')
                        print('Local min(<global min) is ' + str(rescaled) + ' Neural Network model should be okay to continue.')
                    else:
                        print('absolute min max boundary is -2.8 +2.8')
                        print('Local min(>global min) is ' + str(rescaled) + ' Neural Network model might need to be reset with a new Global Minimum and Maximum.')

                # check maximum higher bound
                if localMinMax[nCol][2] > minMax[nCol][2]:
                    outsideMaxPercent = ((localMinMax[nCol][2] - minMax[nCol][2])/(minMax[nCol][2]-minMax[nCol][1]))*100
                    print('Maximum minMax limit operating ' +str(outsideMaxPercent) +' percent above global set maximum.')
                    #absolute min max boundary is -2.8 +2.8
                    rescaled = ((localMinMax[nCol][2] - ((minMax[nCol][2] + minMax[nCol][1]) / 2)) / ((minMax[nCol][2] - minMax[nCol][1]) / 2)) * self.tanhConstant
                    if rescaled <= 2.8:
                        print('absolute min max boundary is -2.8 +2.8')
                        print('Local max(>global max) is ' + str(rescaled) + ' Neural Network model should be okay to continue.')
                    else:
                        print('absolute min max boundary is -2.8 +2.8')
                        print('Local max(>global max) is ' + str(rescaled) + ' Neural Network model might need to be reset with a new Global Minimum and Maximum.')

        elif minMaxv2 != [] and localMinMaxv2 != []:
            for nCol in range(0,len(minMaxv2)):
                # check minimum lower bound
                if localMinMaxv2[nCol][1]<minMaxv2[nCol][1]:
                    outsideMinPercent = ((minMaxv2[nCol][1]-localMinMaxv2[nCol][2]) / (minMaxv2[nCol][2] - minMaxv2[nCol][1])) * 100
                    print('Minimum minMax limit operating ' + str(outsideMinPercent) + ' percent below global set minimum.')
                    # absolute min max boundary is -2.8 +2.8
                    rescaled = ((localMinMaxv2[nCol][1] - ((minMaxv2[nCol][2] + minMaxv2[nCol][1]) / 2)) / ((minMaxv2[nCol][2] - minMaxv2[nCol][1]) / 2)) * self.tanhConstant
                    if rescaled >= -2.8:
                        print('absolute min max boundary is -2.8 +2.8')
                        print('Local min(<global min) is ' + str(rescaled) + ' Neural Network model should be okay to continue.')
                    else:
                        print('absolute min max boundary is -2.8 +2.8')
                        print('Local min(>global min) is ' + str(rescaled) + ' Neural Network model might need to be reset with a new Global Minimum and Maximum.')

                # check maximum higher bound
                if localMinMaxv2[nCol][2] > minMaxv2[nCol][2]:
                    outsideMaxPercent = ((localMinMaxv2[nCol][2] - minMaxv2[nCol][2])/(minMaxv2[nCol][2]-minMaxv2[nCol][1]))*100
                    print('Maximum minMax limit operating ' +str(outsideMaxPercent) +' percent above global set maximum.')
                    #absolute min max boundary is -2.8 +2.8
                    rescaled = ((localMinMaxv2[nCol][2] - ((minMaxv2[nCol][2] + minMaxv2[nCol][1]) / 2)) / ((minMaxv2[nCol][2] - minMaxv2[nCol][1]) / 2)) * self.tanhConstant
                    if rescaled <= 2.8:
                        print('absolute min max boundary is -2.8 +2.8')
                        print('Local max(>global max) is ' + str(rescaled) + ' Neural Network model should be okay to continue.')
                    else:
                        print('absolute min max boundary is -2.8 +2.8')
                        print('Local max(>global max) is ' + str(rescaled) + ' Neural Network model might need to be reset with a new Global Minimum and Maximum.')

    def test(self):

        columns = []
        FEATURE_REQUEST = 'AUDUSD'

        columns.append('Rate')  # o
        pdData = self.DSQuandl.getFeatData(FEATURE_REQUEST)  # in TDelta Lots


        npData = pdData.as_matrix(columns)
        #np.gradient()

        print ('npData: ' + str(npData))

        minVal = np.min(npData)  # #find biggest + rate of change within TDelta
        maxVal = np.max(npData)  # #find biggest - rate of change within TDelta


        print ('minVal: ' + str(minVal))
        print ('maxVal: ' + str(maxVal))

    ###HISTORICAL
    def __init__(self,MODE='DIFF', isRangeZeroToOne=True, isTanhOn=True):
        self.MODE               = MODE
        if isRangeZeroToOne == True:
            self.featRangeMode      = 0# 0 = 0-1, 1 =-1,1
            self.isTanhOn           = False
        else:
            print('featRangeMode -1 to +1')
            self.featRangeMode = 1  # 0 = 0-1, 1 =-1,1
            self.isTanhOn = isTanhOn
        self.tanhConstant       = 2             # SigmoidMax

        self.MinMaxList         = []    #Global min max
        self.MinMaxListv2       = []    #Local difference min max
        self.MinMaxListv3       = []  # Global min max
        self.MinMaxListv4       = []  # Global min max

        self.LocalMinMaxList    = []    #Local min max current time series
        self.LocalMinMaxListv2  = []    #Local difference current time series
        self.LocalMinMaxListv3  = []  # Local min max current time series
        self.LocalMinMaxListv4  = []  # Local min max current time series
        self.firstRowVals       = []
        self.FirstRun           = True       #first lot of data is used to set up minandmax ranges
        self.df_diff_orig       = pd.DataFrame()





#NORM = DataNormalisation()
#NORM.test()