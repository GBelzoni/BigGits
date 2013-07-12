'''
Created on Jun 25, 2013

@author: phcostello
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.core.numeric import dtype
import time
import pickle
import datetime

import strategy_tester.market_data as md
import strategy_tester.trade as td
import strategy_tester.Portfolio as pf
import strategy_tester.TradeStrategyClasses as tsc
import strategy_tester.ResultsAnalyser as ra
import DataHandler.DBReader as dr
import os
os.chdir('/home/phcostello/Documents/workspace/FinancePython')

if __name__ == '__main__':
    
    
    
    def SimpleMA(seriesName, sig_index, short_win, long_win):
        
        #Read in data and setup object
        dbpath = "/home/phcostello/Documents/Data/FinanceData.sqlite"
        the_data_reader = dr.DBReader(dbpath)
        series_data = the_data_reader.readSeries(seriesName)
        
        d1 = datetime.datetime(2009,6,1).date()
        d2 = datetime.datetime(2012,1,1).date()
        #d2 = datetime.datetime.now().date()
        
        series_data = series_data[d1:d2]
        
        #Filter on index we want to look at, e.g 'Open', 'Adj_Close'
        series_data = pd.DataFrame(series_data[sig_index])
        
        #Construct data object for pairs trading strat
        #sma_md = md.simple_ma_md(series_data)
        sma_md = md.exponential_ma_md(series_data)
        
#        sma_md.generateTradeSig(  seriesName = seriesName, 
#                                 sig_index=sig_index, 
#                                 short_win=short_win, 
#                                long_win=long_win)
        
        sma_md.generateTradeSig(  seriesName = seriesName, 
                                 sig_index=sig_index, 
                                 short_win=short_win, 
                                long_win=long_win)
        
        
        #Setup portfolio
        simpleEmptyTrade = td.TradeEquity(seriesName, 
                                          notional=0, 
                                          price_series_label=sig_index)
        
        port = pf.Portfolio("portfolio", cashAmt=100)
        port.add_trade(simpleEmptyTrade)
        #No more trade types
        port.fixed_toggle()
        
        #Setup Strategy
        strat = tsc.MA_Trade_Strategy()
        
        #return pairsStrat
    
        tic = time.clock()
        strat.run_strategy(sma_md,port)
        toc = time.clock()
        print "strategy took {} seconds to run".format(toc - tic)
        outfile = open("pickled_sma2.pkl", 'wb')
        pickle.dump(strat,outfile)
        outfile.close()
        
    def Analysis(pickled_strategy):
    
        pck_file = open(pickled_strategy)
        strategy = pickle.load(pck_file)
        analyser = ra.ResultsAnalyser(strategy,referenceIndex=None)
        
        #Print trade summary
        print 'Sortino Roatio', analyser.sortino_ratio()
        print 'Sharpe over series', analyser.sharpe_ratio()
        #analyser.summary()
        
        #Plot results
        fig = plt.figure()
        ax1= fig.add_subplot(3,1,1)
        ax2 = fig.add_subplot(3,1,2)
        ax3= fig.add_subplot(3,1,3)
        
        print strategy.market_data.info()
        #sig_index = strategy.market_data.sig_index
        sig_index='Adj_Close'
        
        strategy.market_data[[sig_index,'MAs','MAl']].plot(ax=ax1)
        ax1.legend().set_visible(False)
        #print strategy.result['Value'].head()
        #Plot series
        series = strategy.market_data['Adj_Close']
        val = strategy.result['Value']
        val.plot(ax=ax1)
        
        indSeries = series# - series[0] + 100
        pd.DataFrame(indSeries).plot(ax=ax2)
#        
        
        
        cumperiods = 1
        rets = analyser.get_returns(cumperiods= cumperiods)
        maxprob =rets['Portfolio'][ rets['Portfolio'] == rets.max()['Portfolio']]
        rets.plot(ax=ax3)
        
        fig2 = plt.figure()
        val2 = strategy.result['Value'][ maxprob.index[0] - datetime.timedelta(10) : maxprob.index[0] + datetime.timedelta(10)]
        #print val2
        #plt.show()
#        
#        dd = analyser.draw_downs()
#        #print dd['Drawdown']
#        ts = - dd['Drawdown']
#        ts.plot(ax=ax3)
        plt.show()
        
    SimpleMA(seriesName = 'AAPL', sig_index='Adj_Close', short_win = 20, long_win = 40)
    Analysis("pickled_sma2.pkl")
    