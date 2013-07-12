
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.core.numeric import dtype
import time
import pickle
import datetime
import logging
import os
import sys
sys.path.append('/home/phcostello/Documents/workspace/FinancePython')
sys.path.append('/home/phcostello/Documents/workspace/FinancePython/DataHandler')

import strategy_tester.market_data as md
import strategy_tester.trade as td
import strategy_tester.Portfolio as pf
import strategy_tester.TradeStrategyClasses as tsc
import strategy_tester.ResultsAnalyser as ra
import DataHandler.DBReader as dr

import timeit

def grid_tester():
    
    ##################Initialise##########################
    #Read in data and setup object
    dbpath = "/home/phcostello/Documents/Data/FinanceData.sqlite"
    seriesName = 'AUDUSD'
    #sig_index ='Adj_Close' for yahoo
    sig_index ='Rate' #for FX
    
    
    projectpath = '/home/phcostello/Documents/workspace/FinancePython/'
    os.chdir(projectpath)
    
    #import data
    the_data_reader = dr.DBReader(dbpath)
    series_data = the_data_reader.readSeries(seriesName)
    
    
    
    #Set max date and min data
    d1 = datetime.datetime(2010,1,2).date()
    d2 = datetime.datetime(2013,1,1).date()
    series_data = series_data[d1:d2]
    #Filter on index we want to look at, e.g 'Open', 'Adj_Close'
    
    series_data = pd.DataFrame(series_data[sig_index])
    
    #setup logging
    logfile = projectpath + 'logs/GridTester_ma.txt'
    logging.basicConfig(filename= logfile, filemode='w', level = logging.ERROR)
    
    
    ##################Make par Table##########################
    short_wins = np.arange(5,45,10)
    long_wins = np.array([10,30,50,100])
    strat_parameters = [ [sw, lw] for sw in short_wins for lw in long_wins if sw < lw]
    
    
    ##################strat train pars + Dates##########################
    
    #Make dates
    periodLength = 30
    print "Period Length is", periodLength, "days"
    num_periods = (d2-d1).days/periodLength
    num_periods
    periods = [ d2 - x*datetime.timedelta(periodLength) for x in range(0,num_periods)]
    periodRanges = [ list(rng) for rng in zip( periods[1:],periods[:-1])] #quick way of making consecutive ranges
    
    #Make table of parameters + dates
    parameter_table = [ pars + date_range for pars in strat_parameters for date_range in periodRanges]
    len(parameter_table)
    
    ##################Make Res Table##########################
    
    results_list = ['returns',\
                    'volatility',\
                    'sharpe', \
                    'sortino', \
                    'maxDD_level',\
                    'DD_level_start',\
                    'DD_level_end',\
                    'DD_level_length',\
                    'maxDD_duration',\
                    'DD_duration_start',\
                    'DD_duration_end',\
                    'DD_duration_length',\
                    'ETL_5',\
                    'ETL_1']
                    
                    
                    
    results_tables = list(parameter_table) #copy parameter table
    results_tables = pd.DataFrame(results_tables)
    results_tables.columns = ['short_win','long_win','start_evaluation','end_evaluation']
    #Add zero columns for results
    for rstlbl in results_list:
        results_tables[rstlbl] = 0
        
    results_tables
    results_tables.head()
    
    
    ##################Test strat on pars in Table##########################
    
    ##################setupStrat(par) -define function##########################
    #%cpaste
    def setup_strategy(pars):
        short_win = pars[0]
        long_win = pars[1]
        
        sma_md = md.exponential_ma_md(series_data)
        sma_md.generateTradeSig( seriesName = seriesName, 
                                 sig_index=sig_index, 
                                 short_win=short_win, 
                                long_win=long_win)
        #Setup portfolio
        simpleEmptyTrade = td.TradeEquity(seriesName, 
                                          notional=0, 
                                          price_series_label=sig_index)
        
        port = pf.Portfolio("portfolio", cashAmt=1)
        port.add_trade(simpleEmptyTrade)
        #No more trade types
        port.fixed_toggle()
    
        #Setup Strategy
        strat = tsc.MA_Trade_Strategy()
        
        return sma_md, port, strat
    
    #--
    
    ##################strat.runStrat##########################
    def runStrat(strat_parameters,results_tables):
    
        rownum=0
        
        for pars in strat_parameters:
        
            
            market_data, portfolio, strategy = setup_strategy(pars)
            tic = time.clock()
            strategy.run_strategy(market_data , portfolio)
            toc = time.clock()
            print "strategy took {} seconds to run".format(toc - tic)
        
        
        ##################GatherResults##########################
        
            analyser = ra.ResultsAnalyser(strategy,referenceIndex=None)
            
            ##################Add results for dates##########################
            for k in range(0,num_periods):
                try:
                    row = results_tables.iloc[rownum] 
                    #reload(ra)
                    analyser.set_results_range(row['start_evaluation'], row['end_evaluation'],make_positive=True)
                    
                    #Need to fix drawdowns have drawdown equal maxdrawdown always
                    #Need length of maxdd to be where maxx dd occured
                    
                    
                    row['returns']= analyser.get_result().iat[-1,0]/analyser.get_result().iat[0,0] - 1 
                    row['volatility']= analyser.get_volatility()
                    row['sharpe']=analyser.sharpe_ratio()
                    row['sortino']=analyser.sortino_ratio()
                    
                    ##Drawdowns
#                    maxDD_lve = analyser.max_draw_down_magnitude()
#                    maxDD_dur = analyser.max_draw_down_timelength()
#                    row['maxDD_level']=maxDD_lve.loc['Drawdown']
#                    row['DD_level_start']=maxDD_lve.loc['StartDD']
#                    row['DD_level_end']=maxDD_lve.loc['EndDD']
#                    row['DD_level_length']=maxDD_lve.loc['LengthDD']
#                    row['maxDD_duration']=maxDD_dur.loc['Drawdown'][0]
#                    row['DD_duration_start']=maxDD_dur.loc['StartDD'][0]
#                    row['DD_duration_end']=maxDD_dur.loc['EndDD'][0]
#                    row['DD_duration_length']=maxDD_dur.loc['LengthDD'][0]
#                    row['ETL_5']=analyser.etl(0.05)
#                    row['ETL_1']=analyser.etl(0.01)
                    
#                    ##Timing ops
#                    t0 = time.time()
#                    row['returns']= analyser.get_result().iat[-1,0]/analyser.get_result().iat[0,0] - 1 
#                    print analyser.get_result().iat[-1,0]/analyser.get_result().iat[0,0]
#                    t1 = time.time()
#                    print 'returns time', t1 - t0
#                    
#                    row['volatility']= analyser.get_volatility()
#                    t2 = time.time()
#                    print 'volatility time', t2 - t1
#                    
#                    row['sharpe']=analyser.sharpe_ratio()
#                    t3 = time.time()
#                    print 'sharpe time', t3 - t2
#                    
#                    row['sortino']=analyser.sortino_ratio()
#                    t4 = time.time()
#                    print 'sortino time', t4 - t3
#                    
#                    row['maxDD_level']=maxDD_lve.loc['Drawdown']
#                    t5 = time.time()
#                    print 'duration level time', t5 - t4
#                    
#                    row['DD_level_start']=maxDD_lve.loc['StartDD']
#                    t6 = time.time()
#                    print 'duration start lookup time', t6 - t5
#                    
#                    row['DD_level_end']=maxDD_lve.loc['EndDD']
#                    t7 = time.time()
#                    print 'duration level end lookup time', t7-t6
#                    
#                    row['DD_level_length']=maxDD_lve.loc['LengthDD']
#                    t7 = time.time()
#                    print 'duration level length lookup time', t7-t6
#                    
#                    row['maxDD_duration']=maxDD_dur.loc['Drawdown'][0]
#                    t8 = time.time()
#                    print 'duration length max time', t8 -t7
#                    
#                    row['DD_duration_start']=maxDD_dur.loc['StartDD'][0]
#                    t9 = time.time()
#                    print 'duration lenth start lu time', t9 - t8
#                    
#                    row['DD_duration_end']=maxDD_dur.loc['EndDD'][0]
#                    t10 = time.time()
#                    print 'durations end time', t10 -t9
#                    
#                    row['DD_duration_length']=maxDD_dur.loc['LengthDD'][0]
#                    t11 = time.time()
#                    print 'duration lenth lengtht lu time', t11 - t10
#                    
#                    row['ETL_5']=analyser.etl(0.05)
#                    t12 = time.time()
#                    print 'etl time', t12 - t11
#                    
#                    row['ETL_1']=analyser.etl(0.01)
                    
                    
                except Exception as e:
                    print "Exception", e
                finally:
                    
                    rownum +=1
                
                break
            break
    tic = time.clock()
    runStrat(strat_parameters,results_tables)
    toc = time.clock()
    print "All strategies took {} seconds to run".format(toc - tic)
    
    ##################output results##########################
    results_tables.to_csv('Results_AllOrdinaries.csv', index=False)
    
    #################$ Plot for one set of parameters ########################
    pars = [5,50]
    #print pars, tstart, tend
    market_data, portfolio, strategy = setup_strategy(pars)
    strategy.run_strategy(market_data,portfolio)
    analyser = ra.ResultsAnalyser(strategy,referenceIndex=None)
    
    fig =plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    
    portfolioVal = analyser.get_result()
    portfolioVal['Value'].plot(ax=ax1)
    
    trades_at = portfolioVal[portfolioVal['Signal'].isin(['buy','sell'])]
    
    #Putting trade bars on
    for dt in trades_at.index:
        plt.axvline( dt, ymin =0, ymax =1 )
    
    
    strategy.market_data.loc[portfolioVal.index].plot(ax=ax2)
    plt.subplot(2,1,1)
    plt.title(seriesName)
    plt.show()
    
if __name__ == '__main__':
    
    grid_tester()















