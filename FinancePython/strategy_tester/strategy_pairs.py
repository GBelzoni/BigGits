'''
Created on Jul 7, 2013

@author: phcostello
'''

from BoilerPlate import *

import copy

from TradeStrategyClasses import Reversion_EntryExitTechnical
from market_data import pairs_md

class Strategy_Pairs(Reversion_EntryExitTechnical):
    '''
    Basically a wrapper for the reversion strategy
    using pairs marked data
    '''

    def __init__(self,
                 series_x_label,
                 series_y_label, 
                 window_length,
                 scaling_initial = 1, 
                 intercept_initial = 0,
                 entry_scale = 2 ,
                 exit_scale = 0, ):
        '''
        Constructor
        '''
        #initialise super
        Reversion_EntryExitTechnical.__init__(self)
        
        self.series_x_label = series_x_label
        self.series_y_label = series_y_label
        self.window_length = window_length
        self.scaling = scaling_initial
        self.intercept = intercept_initial
        self.entry_scale = entry_scale
        self.exit_scale = exit_scale
        
    def fit(self, data):
        
        #drop na data so that can fit ols
        md = pairs_md(data.dropna(), self.series_x_label, self.series_y_label)
        md.fitOLS()
        self.scaling = md.results.params[self.series_x_label]
        self.intercept = md.results.params['const']
    
    def run_strategy(self, market_data, portfolio):
        
        plot_flag = False
        
        if (pd.isnull(market_data)).any()[0]:
           
           raise ValueError("Series XY = {0}, {1} do not have equal date range".format(self.series_x_label, self.series_y_label)) 
        
        reg_params = [self.intercept, self.scaling]
        
        md = pairs_md(market_data, 
                      xInd = self.series_x_label,
                      yInd = self.series_y_label)
        
        
        md.generateTradeSigs(windowLength = self.window_length,
                              entryScale= self.entry_scale, 
                              exitScale = self.exit_scale, 
                              reg_params = reg_params
                              )
        
        
        Reversion_EntryExitTechnical.run_strategy(self, 
                                                  md,
                                                  portfolio)
        
        
        if plot_flag:
            md.plot_spreadAndSignals()
            self.result['Value'].plot()
            

if __name__ == '__main__':
    
    def test_pairs_strat():
        
        #prepare data
        import DataHandler.DBReader as dbr
        from strategy_tester.market_data import pairs_md
        from strategy_tester.trade import TradeEquity
        from strategy_tester.Portfolio import Portfolio
        
        dbpath = "/home/phcostello/Documents/Data/FinanceData.sqlite"
        dbreader = dbr.DBReader(dbpath)
        SP500 = dbreader.readSeries("SP500")
        BA = dbreader.readSeries("BA")
        dim = 'Adj_Close'
        SP500AdCl = SP500[dim]
        BAAdCl = BA[dim]
        dataObj = pd.merge(pd.DataFrame(BAAdCl), pd.DataFrame(SP500AdCl), how='inner',left_index = True, right_index = True)
        dataObj.columns = ['y','x']
        
        
        
        #Setup portfolio
        trade_equity_spread = TradeEquity('spread', 
                                          notional=0, 
                                          price_series_label='spread')
        
        port = Portfolio("portfolio", cashAmt=1)
        port.add_trade(trade_equity_spread)
        #No more trade types
        port.fixed_toggle()
        
        strat = Strategy_Pairs(series_x_label = 'x', 
                               series_y_label = 'y', 
                               window_length = 20, 
                               scaling_initial= 1, 
                               entry_scale = 1.0, 
                               exit_scale = 0.5)
        
        dateIndexlen = len(dataObj.index)
        startDate_train = datetime.date(2012,12,30)
        endDate_train = datetime.date(2013,6,30)
        
        startDate_test = datetime.date(2012,7,1)
        endDate_test = datetime.date(2012,12,30)
        
        
        trainData = dataObj.loc[startDate_train:endDate_train]
        testData = dataObj.loc[startDate_test:endDate_test]
        
        strat.fit(trainData)
        #Make sure to make a copy of the initialis portfolio
        #Or when you rerun the portfolio it will be the end of the previous
        #run
        portTrain = copy.deepcopy(port)
        strat.run_strategy(market_data = trainData,
                            portfolio = portTrain)
        
        portTest = copy.deepcopy(port)
        strat.run_strategy(market_data = testData,
                            portfolio = portTest)
#
#        
#        strat.fit(testData)
#        portTest2 = copy.deepcopy(port)
#        strat.run_strategy(market_data = testData,
#                            portfolio = portTest2)

        plt.show()
        
 #       strat.result.to_csv('Results/pairsmd.csv')
        
    test_pairs_strat()
        
    