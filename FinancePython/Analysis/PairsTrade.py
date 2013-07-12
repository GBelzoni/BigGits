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

if __name__ == '__main__':
    
    def PairTradeSP500(stdWindowLength, entryScale, exitScale):
        
        #Read in data and setup object
        dbpath = "/home/phcostello/Documents/Data/FinanceData.sqlite"
        the_data_reader = dr.DBReader(dbpath)
        SP500 = the_data_reader.readSeries("SP500")
        BA = the_data_reader.readSeries("BA")
        
        dim = 'Adj_Close' #Choose dim to analyse
        SP500AdCl = SP500[dim]
        BAAdCl = BA[dim]
    #        print SP500AdCl.head()
    #        print BAAdCl.head()
        dataObj = pd.merge(pd.DataFrame(BAAdCl), pd.DataFrame(SP500AdCl), how='inner',left_index = True, right_index = True)
        dataObj.columns = ['y','x']
        
        #Construct data object for pairs trading strat
        pmd = md.pairs_md(dataOb=dataObj,xInd='x',yInd='y')
        maxdate = datetime.date(2013,1,1)
        pmd.core_data = pmd.core_data[:maxdate]
        pmd.fitOLS()
        pmd.generateTradeSigs(stdWindowLength, entryScale, exitScale)
        
        #Setup portfolio
        spreadTrade = td.TradeEquity("spread", notional=0, price_series_label="spread")
        port = pf.Portfolio("portfolio", cashAmt=100)
        port.add_trade(spreadTrade)
        #No more trade types
        port.fixed_toggle()
        
        #Setup Strategy
        pairsStrat = tsc.Reversion_EntryExitTechnical(market_data=pmd, portfolio=port, initial_time_index=1)
        
        #return pairsStrat
    
        tic = time.clock()
        pairsStrat.run_strategy()
        toc = time.clock()
        print "strategy took {} seconds to run".format(toc - tic)
        outfile = open("pickled_pairs.pkl", 'wb')
        pairs_strategy_run = pickle.dump(pairsStrat,outfile)
        outfile.close()
        
    def Analysis(pickled_strategy):
    
        pck_file = open(pickled_strategy)
        strategy = pickle.load(pck_file)
        analyser = ra.PairTradeAnalyser(strategy,referenceIndex=None)
        SP500 = strategy.market_data.core_data['x']
        indSP = SP500/SP500[0]*100
        indSP.index = pd.to_datetime(indSP.index)
        
        #Print trade summary
        print analyser.sortino_ratio()
        analyser.summary()
              
        
        #Plot results
        fig = plt.figure()
        ax1= fig.add_subplot(3,1,1)
        ax2 = fig.add_subplot(3,1,2)
        ax3= fig.add_subplot(3,1,3)
       
        strategy.market_data.core_data[['spread','entryUpper','exitUpper','entryLower','exitLower']].plot(ax=ax1)
        ax1.legend().set_visible(False)
        #print strategy.result['Value'].head()
        strategy.result['Value'].plot(ax=ax2)
        pd.DataFrame(indSP).plot(ax=ax2)
        cumperiods = 252
        rets = analyser.get_returns(cumperiods= cumperiods)
        #print rets.iloc[100:200]
        rets.plot(ax=ax3)
        
        #plt.show()
        
        dd = analyser.draw_downs()
        #print dd['Drawdown']
        ts = - dd['Drawdown']
        #ts.plot(ax=ax3)
        #plt.show()
        
    #PairTradeSP500(stdWindowLength=50, entryScale=2.5, exitScale=0.5)
    Analysis("pickled_pairs.pkl")
    