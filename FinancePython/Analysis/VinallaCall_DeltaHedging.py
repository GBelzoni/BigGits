'''
Created on Jun 23, 2013

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
import DataHandler.GenerateData as gd

def DeltaHedgeVanillaCallEg():
    
    steps = 3000
    stepsize = 1.0/steps
    r = 0.05
    dividend = 0.0 
    vol = 0.2
    S0 = 50.0
    t0 = 0.0
    expiry = 1.0
    K = 50.0
    
    #setup market data#
    #Generate Series
    rseries = steps*[r]
    dividendseries = steps*[dividend]
    volseries = steps*[vol]
    underlyingSeries = gd.GenerateLogNormalTS(S0, mu=0.03, covariance=vol, stepsize=stepsize,steps=steps-1).get_data()
    
    data2 = [rseries,dividendseries,volseries, underlyingSeries]
    data2 = np.array(data2)
    data2.shape
    data2 = data2.transpose()
    data2[1,:]
    
    columns = ['rate','dividend','vol','underlying']
    data = pd.DataFrame(data2, columns = columns)
    
    data.index = list(np.arange(0,steps,dtype='float64')/steps)
    md1 = md.market_data(data)
    
    #need to add to self to use in test functions
    md_slice = md.market_data_slice(md1,time_index=0)
    md_slice.data
    
    tradeUnderlying = td.TradeEquity('underlying',
                                      notional= 0,
                                      price_series_label = 'underlying')
    
    tradeCall = td.TradeVanillaEuroCall(name = "Call",
                                        notional = 0,
                                        strike = K,
                                        expiry = expiry)
                                        
    price = tradeCall.price(md_slice)
    print "price = ", price
    delta = tradeCall.delta(md_slice)   
    print "delta = ", delta
    
    #Setup portfolio
    #First initialise trade type but empty portfolio
    port1 = pf.Portfolio("port1")
    port1.add_trade(tradeUnderlying)
    port1.add_trade(tradeCall)
    
    #Second initialise starting value
    initPort = {'Call':1} 
    port1.adjustNotional(initPort)
    delta = tradeCall.delta(md_slice) 
    print "delta", delta
    trade = {'underlying':-delta}
    port1.adjustNotional(trade)
    port1Slice = pf.PortfolioSlice(portfolio = port1, 
                                market_data= md1, 
                                time_index = 0)
    
    initHedgPort = {'Call':1, "underlying":-delta}
    port1Slice.adjustCash(initHedgPort)
    
    #addsome cash
    MoreCash = {'Cash':1}
    port1.adjustNotional(MoreCash)
    
    prt1Val = port1Slice.value()
    print "Portfolio Value" , prt1Val
    
    prt1Del = port1Slice.delta()
    print "Portfolio Del" , prt1Del 
    
    ts_deltaHedge = tsc.Delta_Hedging(market_data = md1, 
                                  portfolio = port1, 
                                  initial_time_index = 0,
                                  stepsize = stepsize)
    
    ts_deltaHedge.run_strategy()        
    outfile = open('VanillaCallDelta_strat.pkl','wb')
    pickle.dump(ts_deltaHedge,outfile)
    outfile.close()
    print ts_deltaHedge.result.head(20)
    print ts_deltaHedge.result.tail(20)
    print ts_deltaHedge.portfolio.get_notional()
    
def SimpleAnalysis(strategy):

    pck_file = open(strategy)
    strategy = pickle.load(pck_file)
    pck_file.close()
    analyser = ra.ResultsAnalyser(strategy,referenceIndex=None)
    
    #Print trade summary
    analyser.summary()

if __name__ == '__main__':
    
    DeltaHedgeVanillaCallEg()
    SimpleAnalysis('VanillaCallDelta_strat.pkl')
