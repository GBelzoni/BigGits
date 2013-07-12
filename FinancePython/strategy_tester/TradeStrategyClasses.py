'''
Created on Dec 6, 2012

@author: phcostello
'''

import market_data as md
import trade as td
import Portfolio as pf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.core.numeric import dtype
import time
import pickle
import datetime
from pandas.tseries.offsets import DateOffset


class Trade_Strategy(object):
    '''
    classdocs
    '''

    def __init__(self, parameters = None):
        '''
        Constructor
        '''
        self.parameters = parameters
        self.market_data = None
        self.portfolio = None
        self.timeIndex = None
        self.result = None
        
    def upd_signal(self):
        pass
    
    def upd_portfolio(self):
        pass
    
    def run_strategy(self,market_data,portfolio):
        pass

#class Delta_Hedging(Trade_Strategy):
#    
#    #TODO fix market data object so that you don't have to put in increment size in constructor
#    # should read from market data object. Maybe md slice object
#    
#    def __init__(self, market_data, portfolio, initial_time_index, stepsize):
#        '''
#        Constructor
#        '''
#        Trade_Strategy.__init__(self, market_data, portfolio, initial_time_index)
#        self.stepsize = stepsize
#        
#    
#    def upd_signal(self):
#        
#        portSlice = pf.PortfolioSlice(portfolio = self.portfolio, 
#                                      market_data = self.market_data,
#                                      time_index = self.time)
#        
#        delta = portSlice.delta()
#        
#        return delta
#        
#    def upd_portfolio(self):
#        
#        #Update portfolio by making delta neutral
#        delta = self.upd_signal()
#        
#        #Define hedging trade
#        trade = {'underlying' : -delta}
#        
#        #Adjust notional to hedge
#        self.portfolio.adjustNotional(trade)
#        
#        #Adjust cash to reflect heding notional change
#        portSlice = pf.PortfolioSlice(portfolio = self.portfolio, 
#                                      market_data = self.market_data,
#                                      time_index = self.time)
#        portSlice.adjustCash(trade)
#        
#        
#        
#    
#    def run_strategy(self):
#        
#        #Run strategy making sure to inflate notional each step
#        timeInd = self.time
#        maxLoop = len(self.market_data.core_data)
#        num_results = maxLoop - timeInd 
#            
#        self.result = np.zeros((num_results,),dtype=[('Time','f4'),('Value','f4'),('Signal','f4')])
#        #pd.DataFrame(columns = ('Time','Value','Signal'))
#        
#        
#        for i in range(1,(num_results-1)):
#                     
#            timeInd = self.time
#            signal = self.upd_signal()
#            portfolio_slice = pf.PortfolioSlice(self.portfolio,self.market_data,self.time)
#            md_slice = portfolio_slice.md_slice
#            md_slice.period_length = self.stepsize
#            #print "del Before" , portfolio_slice.delta()
#            
#            time = self.market_data.core_data.index[timeInd]
#            
#            #update portfolio so which delta hedges for this period
#            self.upd_portfolio()
#            #print "del After" , portfolio_slice.delta()
#            #Record result
#            res_row = (time, sum(portfolio_slice.value()),signal)
#            #print res_row
#            self.result[i]  = res_row          
#            
#            
#            #Increase time to next period - make sure to inflate the cash by the interest rate
#            self.portfolio.trades['Cash'].inflate(md_slice)
#            self.time +=1
#            
#        
#        self.result = pd.DataFrame(data = self.result)     
#        
#class Stop_Loss(Trade_Strategy):
#    
#    def __init__(self, market_data, portfolio, initial_time_index, stepsize):
#        '''
#        Constructor
#        '''
#        Trade_Strategy.__init__(self, market_data, portfolio, initial_time_index)
#        self.stepsize = stepsize
#    
#    
#    def upd_signal(self):
#        
#        portSlice = pf.PortfolioSlice(portfolio = self.portfolio, 
#                                      market_data = self.market_data,
#                                      time_index = self.time)
#        
#        delta = portSlice.delta()
#        
#        return delta
#        
#    def upd_portfolio(self):
#        
#        #Update portfolio by making delta neutral
#        delta = self.upd_signal()
#        
#        #upd portfolio
#        name = "Hedge" + str(self.time)
#        thisHedge = td.TradeEquity(name = name,
#                                   notional = - delta, 
#                                   price_series_label = 'underlying')
#        
#        self.portfolio.add_trade(thisHedge)
#        
#        #update cash for trade
#        md_slice = md.market_data_slice(self.market_data, self.time)
#        self.portfolio.trades[0].notional -= thisHedge.value(md_slice)
#        
#    
#    def run_strategy(self):
#        
#        #Run strategy making sure to inflate notional each step
#        timeInd = self.time
#        maxLoop = len(self.market_data.core_data)
#        num_results = maxLoop - timeInd 
#            
#        self.result = np.zeros((num_results,),dtype=[('Time','f4'),('Value','f4'),('Signal','f4')])
#        #pd.DataFrame(columns = ('Time','Value','Signal'))
#        
#        
#        for i in range(1,(num_results-1)):
#                     
#            timeInd = self.time
#            signal = self.upd_signal()
#            portfolio_slice = pf.PortfolioSlice(self.portfolio,self.market_data,self.time)
#            md_slice = portfolio_slice.md_slice
#            md_slice.period_length = self.stepsize
#            #print "del Before" , portfolio_slice.delta()
#            
#            time = self.market_data.core_data.index[timeInd]
#            
#            #update portfolio so which delta hedges for this period
#            self.upd_portfolio()
#            #print "del After" , portfolio_slice.delta()
#            #Record result
#            res_row = (time, sum(portfolio_slice.value()),signal)
#            #print res_row
#            self.result[i]  = res_row          
#            
#            
#            #Increase time to next period - make sure to inflate the cash by the interest rate
#            self.portfolio.trades[0].inflate(md_slice)
#            self.time +=1
#            
#        
#        self.result = pd.DataFrame(data = self.result)
# 
class MA_Trade_Strategy(Trade_Strategy): 
    
    '''
    Implements methods for MA trade strategy
    requires simple_ma market data type with column labels
    which stores series names in self.sigIndex, and creates ma's MAl and MAs
    '''
        
    def upd_signal(self):
        
        #TODO need to put in ma trade signal when no ma existing, ie, do nothing
        #that way we don't have weird issues of when strategy starts
          
        ### This returns the update signal from the current data ###
        t1ind = self.timeIndex
        t0ind = self.timeIndex - 1
        
        t1 = self.market_data.index[t1ind]
        t0 = self.market_data.index[t0ind]
        
        #Previous Moving average vals
        MAs0  = self.market_data.at[t0,"MAs"]    
        MAl0  = self.market_data.at[t0,"MAl"]
     
        #Current MA vals
        MAs1 = self.market_data.at[t1,"MAs"]
        MAl1 = self.market_data.at[t1,"MAl"]
        
        #Check if there is an upcrossing this step
        signal = ""
        if  ( MAs0 < MAl0 ) and ( MAs1 > MAl1 ):
            signal = "buy"
        elif ( MAs0 > MAl0 ) and ( MAs1 < MAl1 ):
            signal = "sell"
        else:
            signal = "hold"
                    
        return(signal)
        
    def upd_portfolio(self, tradeSize):
        
        seriesName = self.market_data.seriesName #This from definition of simple_ma_md
        
        signal = self.upd_signal()
        
        trade = None
        
        if signal == 'buy':
            ammt = 1
        elif signal == 'sell':
            ammt = -1
            
        if not (signal == 'hold'): 
               
            #Define trade
            trade = { seriesName :  ammt*tradeSize}
            
            #Adjust notional for trade
            self.portfolio.adjustNotional(trade)
            
            #Adjust cash to reflect trade
            time = self.market_data.index[self.timeIndex]
            
            self.portfolio.adjustCash(trade, self.market_data, time)

        #increase time index            
        self.timeIndex +=1
        
    def run_strategy(self, market_data, portfolio):
        
        self.market_data = market_data
        self.portfolio = portfolio
        self.timeIndex = 1
        
        maxLoop = len(self.market_data)
        num_results = maxLoop - self.timeIndex 
        tradeSize = 1
        seriesName = self.market_data.seriesName
        #At time long_win, ie first time long ma exists we either go long or short
        #depending relative values of ma's
         
        #Get first trade signal
        sig = self.market_data['MAl'].iloc[self.timeIndex] - self.market_data['MAs'].iloc[self.timeIndex]
        
        #short ma is under long ma then should be short, otherwise long
        if sig > 0:
            longshort = -1
        else:
            longshort = 1
            
        #Do initial trade
        trade = { seriesName :  longshort*tradeSize}
        
        #Adjust notional to hedge
        self.portfolio.adjustNotional(trade)
        self.portfolio.adjustCash(trade, self.market_data, market_data.index[0])
        
        
        upd_signal = self.upd_signal
        upd_portfolio = self.upd_portfolio
        get_val = self.portfolio.value
        
        self.result = np.zeros((num_results,),dtype=[('Time','datetime64'),('Value','f4'),('Signal','a10')])
#        def make_res_tb(market_data, timeIndex):
#            
#            #Initialise results
#            result = np.zeros((num_results,),dtype=[('Time','datetime64'),('Value','f4'),('Signal','a10')])
#        
#            
#            for i in range(0,(num_results)):
#                
#                time = market_data.index[timeIndex]
#                
#                #Make result row and add to table
#                signal = upd_signal()
#                value = get_val(market_data,time)
#                res_row = (time, sum(value),signal)
#                result[i]  = res_row          
#                
#                #Update so next period value reflects updated portfolio
#                upd_portfolio(tradeSize=2)
#                timeIndex +=1
#            
#            return result
        for i in range(0,(num_results)):
                
            time = self.market_data.index[self.timeIndex]
            
            #Make result row and add to table
            signal = upd_signal()
            value = get_val(market_data,time)
            res_row = (time, sum(value),signal)
            self.result[i]  = res_row          
            
            #Update so next period value reflects updated portfolio
            upd_portfolio(tradeSize=2)
            
            
        #self.result = make_res_tb(self.market_data,self.timeIndex)
        #Format to pandas dataframe with Time index
        self.result = pd.DataFrame(data = self.result)
        self.result.set_index('Time',inplace=True) 
        
    
    def print_trades(self):
        print [td.name for td in self.portfolio.trades]
    
    def plot(self):
        
        self.market_data.plot()
  
class Reversion_EntryExitTechnical(Trade_Strategy): 
    
    '''
    Implements a reversion strategy where the marked data has upper and lower
    signals.
    '''
    
    def __init__(self):
        '''
        Constructor
        
        market_data object requires columns - "spread", "entryUpper", "exitUpper", "entryLower", "exitLower"
        Note that the "spread" column indicates just the general series being traded,
        originally implemented for pairs trade spread, hence name
        
        portfolio object requires a trade object with "price_series_label" = "spread"
        '''
        
        Trade_Strategy.__init__(self)
        self.tradeEnteredFlag = False
        self.started = False
        self.timeIndex = 0
        
    def upd_signal(self):
        
        ### This returns the update signal from the current data ###
        time = self.timeIndex
        
        Data0 = self.market_data.iloc[time-1]
        Data1 = self.market_data.iloc[time]
        
        #Previous vals
        spread0  = Data0["spread"]    
        upperEntry0 = Data0["entryUpper"]
        upperExit0 = Data0["exitUpper"]
        lowerEntry0 = Data0["entryLower"]
        lowerExit0 = Data0["exitLower"]
        
        #Current vals
        spread1 = Data1["spread"]
        upperEntry1 = Data1["entryUpper"]
        upperExit1 = Data1["exitUpper"]
        lowerEntry1 = Data1["entryLower"]
        lowerExit1 = Data1["exitLower"]
        
        #Check if there is an upcrossing this step
        #If upcrossing then fade trade, ie sell
        signal = ""
        #Exit Signals
        ##NOTE ORDERING IMPORTANT IF SIGNAL GAP FROM TO CROSS BOTH
        ##EXIT FROM FADIND LONG TO ENTER FADING SHORT WE WANT IT TO EXIT
        if ( spread0 > upperExit0 ) and ( spread1 < upperExit1 ):
            signal = ["Exit","buy"]
        elif ( spread0 < lowerExit0 ) and ( spread1 > lowerExit1 ):
            signal = ["Exit","sell"]
        #Enter Signals
        elif  ( spread0 < upperEntry0) and ( spread1 > upperEntry1 ):
            signal = ["Enter","sell"]
        elif ( spread0 > lowerEntry0 ) and ( spread1 < lowerEntry1 ):
            signal = ["Enter" , "buy"]
        #Do nothing signal
        else:
            signal = ["hold","hold"]
                    
        return(signal)
        
    def upd_portfolio(self, tradeSize):
        
        '''
        THis updates portfolio given signal
        We need a 'spread' trade in the strategy portfolio so we can adjust that notional
        '''
        
        #We enter trade when crossing entry barrier and no trade on
        #We exit signal when crossing exit barrier and trade on
        signal = self.upd_signal()
        entered = self.tradeEnteredFlag
        
        #Check if started
        if self.started == False:
            #If not then check if spread is between entry levels, if not then don't start
            Data = self.market_data.iloc[self.timeIndex]                                  
            spread=Data['spread']
            upperEntry = Data['entryUpper']
            lowerEntry = Data['entryLower']
            if lowerEntry < spread and spread < upperEntry:
                self.started = True
            else:
                #Increase time index
                self.timeIndex +=1
                return
            
        if signal[0] != 'hold':        
            
            if entered == False and signal[0] == "Enter":
                if signal[1] == "sell":
                    td_direction = -1
                elif signal[1] == "buy":
                    td_direction = 1 
                
                #Change entered flag to reflect trade
                self.tradeEnteredFlag = not(self.tradeEnteredFlag)    
                
            elif entered == True and signal[0] == "Exit":
                if signal[1] == "sell":
                    td_direction = -1
                elif signal[1] == "buy":
                    td_direction = 1 
                
                #Change entered flag to reflect trade
                self.tradeEnteredFlag = not(self.tradeEnteredFlag)
            
            else:
                td_direction = 0
            
            #Define trade
            
            trade = { 'spread' :  td_direction*tradeSize}
            
            #Adjust notional for trade
            self.portfolio.adjustNotional(trade)
            
            #Adjust cash to reflect trade
            time = self.market_data.index[self.timeIndex]
            self.portfolio.adjustCash(trade, self.market_data, time)
        
        #Increase time index
        self.timeIndex +=1
        
        
        
        
    def run_strategy(self, market_data, portfolio):
        
        #initialise all run variables
        self.started = False
        self.tradeEnteredFlag = False
        self.timeIndex = 0
        self.market_data = market_data
        self.portfolio = portfolio
        
        maxLoop = len(self.market_data)
        num_results = maxLoop - self.timeIndex 
            
        self.result = np.zeros((num_results,),dtype=[('Time','datetime64'),('Value','f4'),('Signal_bs','a10'),('Signal_enterExit','a10')])
        
        #for optimizing loop speed
        upd_signal = self.upd_signal
        upd_portfolio = self.upd_portfolio
        get_val = self.portfolio.value
        
        for i in range(0,(num_results)):
            
            time = self.market_data.index[self.timeIndex]
            
            #Make result row and add to table
            signal = upd_signal()
            value = get_val(market_data,time)
            res_row = (time, sum(value),signal[0],signal[1])
            self.result[i]  = res_row          
            
            #Update so next period value reflects updated portfolio
            #tradeSize notional = portvalue/ price of spread 
            
            
             #Always bet the whole pot
#            if self.tradeEnteredFlag:
#                trade_size = np.abs(self.portfolio.trades['spread'].notional)
#            elif self.tradeEnteredFlag==False:
#                trade_size = np.absolute(sum(value)/self.market_data['spread'].loc[time]) 
#            
            
            
            ##Constant betsize
            if self.tradeEnteredFlag:
                trade_size = np.abs(self.portfolio.trades['spread'].notional)
            elif self.tradeEnteredFlag==False:
                trade_size = np.abs(100/self.market_data['spread'].loc[time]) 
            
            upd_portfolio(trade_size)
            
            
        #self.result = make_res_tb(self.market_data,self.timeIndex)
        #Format to pandas dataframe with Time index
        self.result = pd.DataFrame(data = self.result)
        self.result.set_index('Time',inplace=True) 
        
        
    def print_trades(self):
        print [td.name for td in self.portfolio.trades]
    
         

