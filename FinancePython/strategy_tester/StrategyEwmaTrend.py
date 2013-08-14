'''
Created on Aug 6, 2013

@author: phcostello
'''


import TradeStrategyClasses as tsc
from market_data import ewma_stdev_entry_exit as emd

from BoilerPlate import *

class EWMA_Trend(tsc.Trade_Strategy):
    '''
    Strategy is to enter when above a scaled stdev away from the MA and exit when
    falls back in
    '''
    def __init__(self, MA_win_len, std_win_len, entryScale, exitScale):
        '''
        Constructor
        '''
        tsc.Trade_Strategy.__init__(self)
        
        self.MA_win_len = MA_win_len
        self.std_win_len = std_win_len
        self.entryScale = entryScale
        self.exitScale = exitScale
        self.position_type = 'out'
        self.trade_size = 1
        self.plot_flag = True
        
    def upd_signal(self):
        
        #Time vars
        seriesName = self.market_data.sig_index
        
        #Check if crossing of entry/exit
        this_entry_crossing = self.market_data.iscrossing(self.timeIndex,
                                                           first_name = seriesName,
                                                           second_name = 'entryUpper')
        
        this_exit_crossing = self.market_data.iscrossing(self.timeIndex,
                                                           first_name = seriesName,
                                                           second_name = 'exitUpper')
        
        if this_entry_crossing == 'up' and self.position_type == 'out':
            #If signal is upcrosses entry signal and not long then should enter long.
            return 'long'
        
#         elif this_entry_crossing == 'down' and self.position_type == 'long':
#             #If signal is upcrosses entry signal and not long then should enter long.
#             return 'out'
#         
#         elif this_exit_crossing == 'up' and self.position_type == 'out':     
#             #If signal downcrosses exit signal and long then should exit
#             return 'long'
#         
        elif this_exit_crossing == 'down' and self.position_type == 'long':     
            #If signal downcrosses exit signal and long then should exit
            return 'out'
        
        else:
            #If neither case then keep same position
            return self.position_type
        
    
    def upd_portfolio(self):
        
        '''Updates portfolio for signal'''
        
        trade_sig = self.upd_signal()
        
        
        if trade_sig != self.position_type and self.timeIndex > 10:
            #IF trade sig is different than position then update to reflect
            #TODO: Need to fix this so tradename reads of porfolio
            if trade_sig == 'long':
                td_direction = 1
#                 trade_size = np.absolute(sum(self.portfolio['Value'])/self.market_data['spread'].loc[time]) #Bet the whole put
                trade_size = np.abs(100/self.market_data[self.market_data.sig_index].loc[0]) #Bet same notional calibrated so value is 100 at time 0 
                
                
            if trade_sig == 'out':
                td_direction = -1
                trade_size = np.abs(self.portfolio.trades['Equity'].notional)
            
            #update state of portfolio to reflect signal
            self.position_type = trade_sig
            
            #Adjust notional for trade
            trade = { 'Equity' :  td_direction*trade_size} #TODO: Need to fix this so tradename reads of porfolio
            self.portfolio.adjustNotional(trade)
            
            #Adjust cash to reflect trade
            time = self.market_data.index[self.timeIndex]
            self.portfolio.adjustCash(trade, self.market_data, time)
        
        #Increase time index
        self.timeIndex +=1
        
        
        
        
    def run_strategy(self,market_data,portfolio):
        
        #initialise all run variables
        self.timeIndex = 0
        self.market_data = emd(market_data,'SP500','Adj_Close')
        
        self.market_data.generateTradeSigs(self.MA_win_len, 
                                           self.std_win_len, 
                                           self.entryScale, 
                                           self.exitScale)
        self.portfolio = portfolio
        maxLoop = len(self.market_data)
        num_results = maxLoop - self.timeIndex 
        
        #for optimizing loop speed
        upd_signal = self.upd_signal
        upd_portfolio = self.upd_portfolio
        get_val = self.portfolio.value
        
        self.result = np.zeros((num_results,),dtype=[('Time','datetime64'),('Value','f4'),('Signal','a10')])
        
        for i in range(0,(num_results)):
            
            time = self.market_data.index[self.timeIndex]
            
            #Update so next period value reflects updated portfolio
            
 

            #Make result row and add to table
            signal = upd_signal()
            value = get_val(market_data,time)
            res_row = (time, sum(value),signal)
            self.result[i]  = res_row          
            upd_portfolio()
             
            
        #Format to pandas dataframe with Time index
        self.result = pd.DataFrame(data = self.result)
        self.result.set_index('Time',inplace=True) 
        
        if self.plot_flag:
            self.market_data.plot_Signals()
#             self.result['Value'].plot()
         
        
    def print_trades(self):
        print [td.name for td in self.portfolio.trades]
    
if __name__ == '__main__':
    
    def test_ewma_strat():
        
        #prepare data
        import DataHandler.DBReader as dbr
        from strategy_tester.trade import TradeEquity
        from strategy_tester.Portfolio import Portfolio
        
        dbpath = "/home/phcostello/Documents/Data/FinanceData.sqlite"
        dbreader = dbr.DBReader(dbpath)
        SP500 = dbreader.readSeries("SP500")
        dim = 'Adj_Close'
        data = SP500
        
        
        startDate_train = datetime.date(2012,12,30)
        endDate_train = datetime.date(2013,6,30)
        dataReduced = data.loc[startDate_train:endDate_train]
        
       
        #Initialise strategy parameters
        strat = EWMA_Trend( MA_win_len=50.0,
                            std_win_len=50.0,
                            entryScale= 0,
                            exitScale= 1)
        
        
        
        #Initialize Portfolio
        
        #Generate Signals on Market data
        
        #Run strategy
        
        
        
        #Setup portfolio
        trade_equity_spread = TradeEquity('Equity', 
                                          notional=0, 
                                          price_series_label='Adj_Close')
        
        port = Portfolio("portfolio", cashAmt=100)
        port.add_trade(trade_equity_spread)
        #No more trade types
        port.fixed_toggle()
       
        strat.plot_flag=True
        strat.run_strategy(market_data = data,
                            portfolio = port)
       
        
        
#         plt.show()
        plt.figure()
        strat.result['Value'].plot()
#         plt.show()
 #       strat.result.to_csv('Results/pairsmd.csv')
        
        from strategy_tester.ResultsAnalyser import ResultsAnalyser
        
        
        ra = ResultsAnalyser(data= strat.result) #Sharpe for strat
        print ra.sharpe_ratio(), ra.get_cumulative_return().iloc[-1,0]
        
        ra2 = ResultsAnalyser(data = SP500,valueIndex='Adj_Close') #Sharpe for SP500
        print ra2.sharpe_ratio(), ra2.get_cumulative_return().iloc[-1,0]
    
        combresdata = pd.merge(strat.result,SP500, how='inner', left_index=True, right_index=True)
        
        ra3 = ResultsAnalyser(data = combresdata,valueIndex='Value',referenceIndex='Adj_Close')
        print ra3.sharpe_ratio(useMarketRef=True), ra3.get_cumulative_return(useReference=True).iloc[-1,0]
    
    test_ewma_strat()
        
   
        