'''
Created on Nov 21, 2012

@author: phcostello
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import pandas.io.sql as psql
import sqlite3
import statsmodels.tsa.stattools as sttl
import statsmodels.tsa as tsa
import statsmodels.api as sm

zoo = importr('zoo')


class market_data(pd.DataFrame):

    ''' Market Data is a pandas Dataframe with methods for adding techinicals fitting models etc '''

    def addSMA(self, fromIndex, addedSeriesName = 'sma', win_length = 1):
        
        self[addedSeriesName] = pd.rolling_mean(self[fromIndex], win_length)
        
    def addEMA(self, fromIndex, addedSeriesName = 'ema', win_length = 1):
        
        self[addedSeriesName] = pd.ewma(self[fromIndex], win_length)
    
    def addSDev(self, fromIndex, addedSeriesName = 'stdev', win_length = 1):
        
        self[addedSeriesName] = pd.rolling_std(self[fromIndex], win_length)
    
    def addBollBandef(self, fromIndex, addedSeriesName = 'BBand', scale=1,win_length = 1):
        
        self[addedSeriesName+'upper'] = scale*pd.rolling_std(self[fromIndex], win_length) + self[fromIndex]
        self[addedSeriesName+'lower'] = -scale*pd.rolling_std(self[fromIndex], win_length) + self[fromIndex]
        
            
#old constructor for constructing from dataframe        
#        try:
#            if isinstance(dataOb, ro.vectors.Matrix) == True:
#                names = ro.r.colnames(dataOb)
#                time_stamp = zoo.index(dataOb)
#                self.core_data = pandas.DataFrame(numpy.array(zoo.coredata(dataOb)), index = time_stamp, columns = names)
#            elif isinstance(dataOb, pandas.DataFrame):
#                self.core_data = dataOb
#            else:
#                raise NameError('Error: Not supported object')
#            
#        except NameError:   
#            raise        
#        
       

class simple_ma_md(market_data):
        
    def generateTradeSig(self , seriesName, sig_index, short_win, long_win):
        
        #Name of series to trade
        self.seriesName = seriesName
        
        #Name of index of data, e.g 'Open' 'Close', 'Adj_Close'
        #put check in to see if sig_index is available
        self.sig_index = sig_index
        
        #Add to data
        self.addSMA(fromIndex = sig_index, 
                    addedSeriesName = 'MAs', 
                    win_length = short_win)
        self.addSMA(fromIndex = sig_index, 
                    addedSeriesName = 'MAl', 
                    win_length = long_win)
        
    def plot(self):
        
        dataLabels = [self.sig_index,
                      'MAl',
                      'MAs']
                        
        self[dataLabels].plot()
        
        
class exponential_ma_md(market_data):

    def __init__(self, dataOb):
            
            #Construct base class data
            market_data.__init__(self, dataOb)   
            self.sig_index = None
            self.seriesName = None
            
            
    def generateTradeSig(self , seriesName, sig_index, short_win, long_win):
        '''Generates long and short exponentially weighted moving averagees '''
        
        #Name of series to trade
        self.seriesName = seriesName
        
        #Name of index of data, e.g 'Open' 'Close', 'Adj_Close'
        #put check in to see if sig_index is available
        self.sig_index = sig_index
        
        #Add to data
        self.addEMA(fromIndex = sig_index, 
                    addedSeriesName = 'MAs', 
                    win_length = short_win)
        self.addEMA(fromIndex = sig_index, 
                    addedSeriesName = 'MAl', 
                    win_length = long_win)
        
    def plot(self):
        
        dataLabels = [self.sig_index,
                      'MAl',
                      'MAs']
                        
        self[dataLabels].plot()
        

class pairs_md(market_data):
    
    def __init__(self, dataOb, xInd, yInd):
        
        #Construct base class data
        market_data.__init__(self, dataOb)   
        self.results = None
        
        #requires dataOb has two indexes that will be pairs in trade
        #define index in data for pairs trade
        self.xInd = xInd
        self.yInd = yInd
        
        
    def fitOLS(self):
        #Run OLS and create residual series
        import statsmodels.api as sm
        x = sm.add_constant(self[self.xInd])
        self.results = sm.OLS(self[self.yInd],x).fit()
        #Optimize get rid of saving results
        
        
    def printSummary(self):
        print self.results.summary()
        
    def adfResids(self):
        
        resid = self.results.resid
        result = sm.tsa.adfuller(resid)
        return result
        
    def generateTradeSigs(self, windowLength, entryScale, exitScale, reg_params = None):
        
        if reg_params == None:
            self['spread'] = self.results.resid
        else:
            self['spread'] = self[self.yInd] - reg_params[1]*self[self.xInd] - reg_params[0]       
        
        self['entryUpper'] = entryScale*pd.rolling_std(self['spread'],windowLength)
        self['entryLower'] = -entryScale*pd.rolling_std(self['spread'],windowLength)
        self['exitUpper'] = exitScale*pd.rolling_std(self['spread'],windowLength)
        self['exitLower'] = -exitScale*pd.rolling_std(self['spread'],windowLength)
    
    def plot_spreadAndSignals(self, ax=None):
        
        dataLabels = ['spread',
                      'entryUpper',
                      'entryLower',
                      'exitUpper',
                      'exitLower']
                        
        plotData = self[dataLabels]
        plotData.plot(ax = ax)
        
        
        
class moving_av_reversion(market_data):
    
    def __init__(self, dataOb, seriesName, sig_index):
        
        ''' 
        Requires the seriesName of trade in initialising portfolio
        sig_index = index of data, e.g 'Open' 'Close', 'Adj_Close'
        length of lags in moving averages
        '''
        #Construct base class data
        market_data.__init__(self, dataOb)
        
        #Name of series to trade
        self.seriesName = seriesName
        
        #Name of index of data, e.g 'Open' 'Close', 'Adj_Close'
        #put check in to see if sig_index is available
        self.sig_index = sig_index
                
        
    def generateTradeSigs(self, MAWin, stdDevWin, entryScale, exitScale):
        
        sig_index = self.sig_index
        
        #generate Moving Ave to revert to
        self.addEMA(sig_index, 'MovingAve', MAWin)
        self['spread'] = self[sig_index] - self['MovingAve']
        
        #Generate Entry/Exit sigs
        spread = self['spread']
        
        #gen moving std of spread
        self.addSDev('spread','entryUpper', stdDevWin)
        self.addSDev('spread','entryLower', stdDevWin)
        self.addSDev('spread','exitUpper', stdDevWin)
        self.addSDev('spread','exitLower', stdDevWin)
        
        self['entryUpper'] *= entryScale
        self['entryLower'] *= -entryScale
        self['exitUpper'] *= exitScale
        self['exitLower'] *= -exitScale
        
        #Signals are distance from moving ave, so add that
        self['entryUpper'] += self['MovingAve']
        self['entryLower'] += self['MovingAve']
        self['exitUpper'] += self['MovingAve']
        self['exitLower'] += self['MovingAve']

        
    def plot_spreadAndSignals(self):
    
        dataLabels = [self.sig_index,
                      'MovingAve',
                      'entryUpper',
                      'entryLower']
#                      'exitUpper',
#                      'exitLower']
#                        
        plotData = self[dataLabels]
        plotData.plot()

        

if __name__ == '__main__':
    
    def test_pairs_md():
    #prepare data
        import DataHandler.DBReader as dbr
        dbpath = "/home/phcostello/Documents/Data/FinanceData.sqlite"
        dbreader = dbr.DBReader(dbpath)
        SP500 = dbreader.readSeries("SP500")
        BA = dbreader.readSeries("BA")
        dim = 'Adj_Close'
        SP500AdCl = SP500[dim]
        BAAdCl = BA[dim]
        dataObj = pd.merge(pd.DataFrame(BAAdCl), pd.DataFrame(SP500AdCl), how='inner',left_index = True, right_index = True)
        dataObj.columns = ['y','x']
        
        pmd = pairs_md(dataOb=dataObj,xInd='x',yInd='y')
        pmd.fitOLS()
        print pmd.results.params
        resid = pmd.results.resid
        resid.plot()
        rllstd = pd.rolling_std(resid,100,min_periods=10)
        rllstd.plot()
        #plt.show()
        
        
        #pmd.printSummary()
        #print pmd.adfResids()
        scale = pmd.results.params['x']
        intercept = pmd.results.params['const']
        reg_params = [intercept, scale]
        pmd.generateTradeSigs(50, entryScale=1.5, exitScale=0, reg_params=reg_params)
        pmd.plot_spreadAndSignals()
        
    def test_MA_reversion_md():
        #prepare data
        import DataHandler.DBReader as dbr
        dbpath = "/home/phcostello/Documents/Data/FinanceData.sqlite"
        dbreader = dbr.DBReader(dbpath)
        SP500 = dbreader.readSeries("SP500")
        dim = 'Adj_Close'
        SP500AdCl = SP500[dim]
        
        marev_md = moving_av_reversion(dataOb = SP500,
                                       seriesName = "SP500",
                                       sig_index =dim)
        
        marev_md.generateTradeSigs(MAWin = 50,
                                   stdDevWin = 20,
                                   entryScale = 2,
                                   exitScale= 0)
       
        marev_md.plot_spreadAndSignals()
        
    
    def test_simple_md():
        
        #read in data for all FTSE100 series
        import pickle
        import os
        os.chdir('/home/phcostello/Documents/workspace/BigGits/FinancePython')
        Alldata = pickle.load(open('pickle_jar/FTSE100_AdjClose.pkl'))
        print 'FTSE100 data info'
        Alldata.info()
        
        #Pick one series
        
        data = Alldata.loc['2012':,'AMEC_plc']
        data = pd.DataFrame(data,columns = ['Adj_Close'])
        
        
        
        #Create market data object - doesn't do much just tells
        #data it now has some addTechnical functions
        md = market_data(data)
        md.plot()
        plt.show()
        
        #Add moving averages and plot
        md.addSMA(fromIndex='Adj_Close', 
                  addedSeriesName = 'Simple_MA', 
                  win_length=20)
        
        print 'FTSE100 data with Simple Moving ave added'
        print md.head()
        md.addEMA(  fromIndex='Adj_Close', 
                  addedSeriesName = 'Exponential_MA', 
                  win_length=20)
        
        
        print 'FTSE100 data with Exponential Moving ave added'
        print md.head()
        md.plot()
        plt.show()
        
        
        #Add bollinger bands and plot
        md.addBollBandef(fromIndex='Exponential_MA', 
                  addedSeriesName = 'BBand_aroundEMA', 
                  scale = 2,
                  win_length=20)
                
        print 'FTSE100 data with BBands added'
        print md.head()
        md.plot()
        plt.show()
        
        
#    test_pairs_md()
#    test_MA_reversion_md()
    test_simple_md()
    
    
        