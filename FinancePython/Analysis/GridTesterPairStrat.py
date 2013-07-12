from BoilerPlate import *
from strategy_tester.market_data import pairs_md
from strategy_tester.trade import TradeEquity
from strategy_tester.Portfolio import Portfolio
from strategy_tester.ResultsAnalyser import ResultsAnalyser
from strategy_tester.strategy_pairs import Strategy_Pairs

#prepare data
import DataHandler.DBReader as dbr



##################setupStrat(par) -define function##########################
#%cpaste
def setup_strategy(pars, dataObj):
    
    #Setup Data
    series_X = pars[0]
    series_Y = pars[1]
    
    entry_scale = pars[3]
    exit_scale = pars[4]
    
    train_start = pars[5]
    train_end = pars[6]
    
    test_start = pars[7]
    test_end = pars[8]
    
    
    #train data
    train_data = dataObj[[series_X,series_Y]].loc[train_start:train_end]
    test_data = dataObj[[series_X,series_Y]].loc[test_start:test_end]
    
    #Setup portfolio
    trade_equity_spread = TradeEquity('spread', 
                                      notional=0, 
                                      price_series_label='spread')
    
    
    port = Portfolio("portfolio", cashAmt=100)
    port.add_trade(trade_equity_spread)
    #No more trade types
    port.fixed_toggle()
    
    #Setup Strategy
    strat = Strategy_Pairs(series_X,
                                  series_Y,
                                  window_length= 20,
                                  scaling_initial = 1,
                                  intercept_initial= 0,
                                  entry_scale = entry_scale,
                                  exit_scale = exit_scale,
                                  )
    
    
    strat.fit(train_data)
    return port, strat, train_data, test_data

##################strat.runStrat##########################
def runStrat(strat_parameters, results_tables, dataObj, outfile):

    rownum = 0
    total_rows = len(strat_parameters)
    
    for pars in strat_parameters:
        
        try:
            portfolio_0, strategy, train_data, test_data = setup_strategy(pars, dataObj)
            
            #Get results on training
            tic = time.clock()
            portfolio = portfolio_0.copy() #Copy initial portfolio so can use in poth train and test
            strategy.run_strategy(train_data , portfolio)
            toc = time.clock()
            print "training strategy took {} seconds to run".format(toc - tic)
            
            analyser = ResultsAnalyser(strategy,referenceIndex=None)
            #Need to fix drawdowns have drawdown equal maxdrawdown always
            #Need length of maxdd to be where maxx dd occured
            row = ['train',
                    pars[0],
                    pars[1],
                    pars[2],
                    pars[3],
                    pars[4],
                    pars[5],
                    pars[6],
                    pars[7],
                    pars[8],
                    analyser.get_result().iat[-1,0]/analyser.get_result().iat[0,0] - 1 ,
                    analyser.get_volatility(),
                    analyser.sharpe_ratio(),
                    analyser.sortino_ratio(),
                    pars[9]]
#            
#            
#            
#            
#            row['runtype']= 'train'
#            row['series_X']= pars[0]
#            row['series_Y']= pars[1]
#            row['rank'] = pars[2]
#            row['entry_scale'] = pars[3]
#            row['exit_scale'] = pars[4]
#            row['train_start'] = pars[5]
#            row['train_end'] = pars[6]
#            row['test_start'] = pars[7]
#            row['test_end'] = pars[8]
#            row['returns']= analyser.get_result().iat[-1,0]/analyser.get_result().iat[0,0] - 1 
#            row['volatility']= analyser.get_volatility()
#            row['sharpe']=analyser.sharpe_ratio()
#            row['sortino']=analyser.sortino_ratio()
##            results_tables = results_tables.append( row, ignore_index=True)
#            row_str = [str(it[1][0]) for it in row.iteritems() ]
            row_str = [str(it) for it in row]
            row_str = ','.join(row_str) + '\n'
            outfile.write(row_str)
            
            
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
            #Get results on training
            tic = time.clock()
            portfolio = portfolio_0.copy() #Copy initial portfolio so can use in poth train and test
            strategy.run_strategy(test_data, portfolio)
            toc = time.clock()
            print "testing strategy took {} seconds to run".format(toc - tic)
            analyser = ResultsAnalyser(strategy,referenceIndex=None)
#            row = pd.DataFrame([rownum]) 
            #Need to fix drawdowns have drawdown equal maxdrawdown always
            #Need length of maxdd to be where maxx dd occured
            row = ['test',
                    pars[0],
                    pars[1],
                    pars[2],
                    pars[3],
                    pars[4],
                    pars[5],
                    pars[6],
                    pars[7],
                    pars[8],
                    analyser.get_result().iat[-1,0]/analyser.get_result().iat[0,0] - 1 ,
                    analyser.get_volatility(),
                    analyser.sharpe_ratio(),
                    analyser.sortino_ratio(),
                    pars[9]]
#                   
#            
#            row['runtype']= 'test'
#            row['series_X']= pars[0]
#            row['series_Y']= pars[1]
#            row['rank'] = pars[2]
#            row['entry_scale'] = pars[3]
#            row['exit_scale'] = pars[4]
#            row['train_start'] = pars[5]
#            row['train_end'] = pars[6]
#            row['test_start'] = pars[7]
#            row['test_end'] = pars[8]
#            row['returns']= analyser.get_result().iat[-1,0]/analyser.get_result().iat[0,0] - 1 
#            row['volatility']= analyser.get_volatility()
#            row['sharpe']=analyser.sharpe_ratio()
#            row['sortino']=analyser.sortino_ratio()
##            results_tables = results_tables.append( row, ignore_index=True)
#            row_str = [str(it[1][0]) for it in row.iteritems() ]
            row_str = [str(it) for it in row]
            row_str = ','.join(row_str) + '\n'
            outfile.write(row_str)
            
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
            rownum +=2
            print rownum/2, 'out of', total_rows, 'done'
            
    return results_tables

def grid_tester(run_strat, do_plots):
        
    ##################Initialise##########################
    #Read in data and setup object
    
#    dbpath = "/home/phcostello/Documents/Data/FinanceData.sqlite"
#    dbreader = dbr.DBReader(dbpath)
#    #SP500 = dbreader.readSeries("SP500")
    #BA = dbreader.readSeries("BA")
    #dim = 'Adj_Close'
    #SP500AdCl = SP500[dim]
    #BAAdCl = BA[dim]
    #dataObj = pd.merge(pd.DataFrame(BAAdCl), pd.DataFrame(SP500AdCl), how='inner',left_index = True, right_index = True)
    #dataObj.columns = ['y','x']
    
    #Read data for all pairs
    #Note have checked top 10 in and constructed data in
    #CheckingSeriesCoint.py
    
    dataObj = pickle.load(open('pickle_jar/FTSE100_AdjClose.pkl'))
    cointAnalysis = pickle.load(open('pickle_jar/results_FTSE100_coint.pkl','rb'))        
    cointAnalysis = pd.DataFrame(cointAnalysis, columns = ['SeriesX_name','SeriesY_name','Adf_pvalue', 'reg_scaling', 'reg_intercept'])
    cointAnalysis = cointAnalysis.sort('Adf_pvalue', ascending=True)
    cointAnalysis.info()
    seriesNames = cointAnalysis[['SeriesX_name','SeriesY_name']]
    dataObj.info()
    
    top10 = seriesNames.iloc[0:10]
    top10['rank']= range(1, len(top10)+1)
    top10.head()
    top10 =top10.values.tolist()
    
    #Filter to what we want
     
    #setup logging
    logfile = 'logs/GridTester_ma.txt'
    logging.basicConfig(filename= logfile, filemode='w', level = logging.ERROR)
    
    
    ##################Make par Table##########################
    #Parameter table rows are
    #[ pairs_labes, strategy_params, train_ranges, test_ranges ]
    
    #series
    series_names = top10[0:6]
    
    #strat pars
    entry_scale = np.arange(1,3.5,0.5)
    exit_scale = np.arange(0.0,3.0,0.5)
    strat_parameters = [ [sw, lw] for sw in entry_scale for lw in exit_scale if sw >= lw]
    
    #Train - test date pars
    dmin = datetime.datetime(2008,1,1)
    num_periods = 10
    period_delta = datetime.timedelta(182)
    #Create range start/end offsets from min data
    train_test_ranges = [ [dmin + i*period_delta, #train start
                         dmin + (i+1)* period_delta, #train end
                         dmin + (i+1)* period_delta, #test start
                         dmin + (i+2)* period_delta] #test end
                         for i in range(0,num_periods)]
    
    #Make table of parameters + dates
    parameter_table = [ series + pars + date_ranges for series in series_names for pars in strat_parameters for date_ranges in train_test_ranges]
    parameter_table = [ parameter_table[i] + [i] for i in range(0,len(parameter_table))]
    print pd.DataFrame(parameter_table).iloc[20:40]
    
    ##################Make Res Table##########################
#    results_list = ['returns',\
#                    'volatility',\
#                    'sharpe', \
#                    'sortino', \
#                    'maxDD_level',\
#                    'DD_level_start',\
#                    'DD_level_end',\
#                    'DD_level_length',\
#                    'maxDD_duration',\
#                    'DD_duration_start',\
#                    'DD_duration_end',\
#                    'DD_duration_length',\
#                    'ETL_5',\
#                    'ETL_1']
    
    colnames =['runtype',\
    'series_X',\
    'series_Y',\
    'rank',\
    'entry_scale',\
    'exit_scale',\
    'train_start',\
    'train_end',\
    'test_start',\
    'test_end',\
    'returns',\
    'volatility',\
    'sharpe',\
    'sortino',\
    'par_rn']
                                
    #results_tables = list(parameter_table) #copy parameter table
    #colnames = ['series_X','series_Y','entry_scale','exit_scale','start_train','end_train','start_test','end_test']
    #colnames = colnames + results_list
    #results_tables = pd.DataFrame(results_tables)
    #results_tables = pd.DataFrame(columns=colnames)
    ##Add zero columns for results
    #for rstlbl in results_list:
    #    results_tables[rstlbl] = 0
#    colnames = ['series_X','series_Y','rank','entry_scale','exit_scale','train_start','train_end','test_start','test_end'] + results_list
    results_tables = pd.DataFrame(columns =colnames)
    print results_tables.head(50)
        
    ##################Test strat on pars in Table##########################
    
    if run_strat:
        outfile = open('Results/dummy.csv','wb')
        header = ','.join(colnames) + '\n'
        outfile.write(header)
      
        tic = time.clock()
        results_tables = runStrat(parameter_table,results_tables,dataObj, outfile)
        toc = time.clock()
        print "All strategies took {} seconds to run".format(toc - tic)
        
        outfile.close()
    
    ##################output results##########################
    
    
#    results_tables.to_csv('Results/Results_pairs_FTSE100.csv', index=False)
    
    #################$ Plot for one set of parameters ########################
    
    #print pars, tstart, tend
    
    #Setup Data
    
    if do_plots:
        this_pars = parameter_table[1221]
        #Setup Data

        print this_pars
        portfolio, strategy, train_data, test_data = setup_strategy(this_pars,dataObj)
        data = train_data
        strategy.run_strategy(data,portfolio)
        analyser = ResultsAnalyser(strategy,referenceIndex=None)
        
        
        fig =plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        
        portfolioVal = analyser.get_result()
        start = 0
        end = 110
        portfolioVal['Value'].iloc[start:end].plot(ax=ax1)
        
#        trades_at = portfolioVal[portfolioVal['Signal_bs'].isin(['buy','sell'])]
#        
#        #Putting trade bars on
#        for dt in trades_at.index:
#            plt.axvline( dt, ymin =0, ymax =1 )
#        
        
        plt.subplot(2,1,1)
        pairs_md(data.loc[portfolioVal.index].iloc[start:end],0,0).plot_spreadAndSignals(ax2)
        
        
        #plt.title(seriesName)
        plt.show()
    
if __name__ == '__main__':
    
    grid_tester(run_strat = False, do_plots = True)
#    grid_tester(run_strat = True, do_plots = False)