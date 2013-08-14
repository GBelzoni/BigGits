'''
Created on Aug 7, 2013

@author: phcostello
'''

from BoilerPlate import *
import TradeStrategyClasses as tsc
from trade import TradeEquity
from Portfolio import Portfolio

class StrategyMSeriesMomentum(tsc.Trade_Strategy):
    '''
    This strategy scores series by momentum, the makes portfolio with given weights
    '''

    def __init__(self, ema_par, lookback=10, lag=0, top_scores_num = 10):
        '''
        Constructor
        '''
        tsc.Trade_Strategy.__init__(self)
        self.ema_par=ema_par
        self.lookback = lookback
        self.lag = lag
        self.top_scores_num = top_scores_num
        
        #This has top weights and rankings, plus series as index
        self.top_series_weights = None
        
    
    def setup_portfolio(self, series_list):
        '''
        Sets up portfolio containing Equity Trades with Names and series related to list
        returns initialised port
        '''
        #Setup portfolio
        trades = [ TradeEquity(series, notional=0,price_series_label=series) for series in series_list]
        
        #Create portfolio
        port = Portfolio("portfolio", cashAmt=100)
        
        for td in trades:
            port.add_trade(td)
            
        #No more trade types
        port.fixed_toggle()
        
        return port
    
    def fit(self,train_data):
        '''
        Scores indexes momentum over period and calculates weighting vector
        stores results in internal parameters
        '''
        
        ema = pd.ewma(train_data, self.ema_par)
        momentum = ema.pct_change(self.lookback)
        ranks = momentum.iloc[-self.lag].rank(ascending=False).order()
        #Pick top 10 and with momentum >0
        topn = self.top_scores_num
        port_series = ranks.iloc[0:topn][ ranks.iloc[0:topn]>0].index
        #Create weightings
        #weights = np.ones(len(port_series))
        ranks_weights = pd.DataFrame(ranks[port_series].copy(),columns =['Ranks'])
        
        #Weight top 10 by shares in ranking
        ranks_weights['weights'] = (ranks_weights['Ranks'].max() - ranks_weights['Ranks']+1)/sum(ranks_weights['Ranks'])
        self.top_series_weights = ranks_weights
        
        
    def upd_signal(self):
        '''
        Nothing much to do here - passive strategy
        '''
        pass
    
    
    def upd_portfolio(self):
        '''
        Weights portfolio by trained weightings in first period
        Passive after
        '''
        #Assume portfolio constructed with series matching those in fitted index
        #print self.portfolio.info()
        
        #If first time caculate notionals to give value weightings
        if self.timeIndex == 1:
            
            time = self.market_data.index[1]
            #Get portfolio val
            value = sum(self.portfolio.value(self.market_data,time))
            top_series = self.top_series_weights.index
            
            trade = None
            for series in top_series:
            #for top series
                
                #Get weights
                weight = self.top_series_weights.loc[series,'weights']
                
                #Calc notional to make weight
                notional = value*weight / self.market_data.at[time,series] 
                
                #put on trades and adjust notional
                trade = { series :  notional}
                
                #Adjust notional for trade
                self.portfolio.adjustNotional(trade)
                
                #Adjust cash for trade
                self.portfolio.adjustCash(trade, self.market_data, time)

        #increase time index            
        self.timeIndex +=1
                
        
    def run_strategy(self, market_data, portfolio):
        '''
        Calcs returns
        '''
    
        self.market_data = market_data
        self.portfolio = portfolio
        self.timeIndex = 1
        
        num_results = len(market_data)-1
        
        upd_signal = self.upd_signal
        upd_portfolio = self.upd_portfolio
        get_val = self.portfolio.value
        
        self.result = np.zeros((num_results,),dtype=[('Time','datetime64'),('Value','f4'),('Signal','a10')])
        for i in range(0,num_results):
            
               
            time = self.market_data.index[self.timeIndex]
            #Make result row and add to table
            signal = "na"
            value = get_val(market_data,time)
            res_row = (time, sum(value),signal)
            self.result[i]  = res_row          
            #Update so next period value reflects updated portfolio
            upd_portfolio()
#             print portfolio.info()
            
            
        #self.result = make_res_tb(self.market_data,self.timeIndex)
        #Format to pandas dataframe with Time index
        self.result = pd.DataFrame(data = self.result)
        self.result.set_index('Time',inplace=True) 
        
    
    def print_trades(self):
        print [td.name for td in self.portfolio.trades]
    
    def plot(self):
        
        self.market_data.plot()

    
if __name__ == '__main__':
    
    def test_strat():
        
        #Already pickled DJIA_AdjClose
        dataObj = pickle.load(open('pickle_jar/DJIA_AdjClose.pkl'))
        #Read DJIA index
        dbr = DBReader()
        DJIA = dbr.readSeries(seriesName = 'DJIA')
        #dataObj['DJIA'] = DJIA
        DJIA = pd.DataFrame(DJIA)#,columns=['DJIA'])
        DJIA.columns = ['DJIA']
        
        series_list = dataObj.columns.values
        
        #setup train/test ranges
        period_length_1 = 90
        period_length_2 = 180 
        start_date = dataObj.index[0]
        end_date = dataObj.index[-1]
        periods_dates = pd.date_range(start_date,end_date,freq = '3BMS')
        periods = [ [periods_dates[i], periods_dates[i+1] ] for i in range(0, len(periods_dates)-1)]
        train_periods = periods[0:-1]
        test_periods = periods[1:]
 
        #Do example
        #setup data
        print len(train_periods)
        period_num = 32
        train_data = dataObj.loc[train_periods[period_num][0]:train_periods[period_num][1]]
        test_data = dataObj.loc[test_periods[period_num][0]:test_periods[period_num][1]]
        
        #Setup strat
        
        #parameters
        top_scores_num = 10 
        ema_par = 50
        lag = 1
        lookback = 10
        
        #Initialise strat
        strat = StrategyMSeriesMomentum( ema_par, lookback, lag, top_scores_num)
        
        #Train - ie pick top X momentum and weights
        strat.fit(train_data)
        
        #Setup portfolio
        series_list = strat.top_series_weights.index.values
        portfolio = strat.setup_portfolio( series_list )
        portfolio_train = portfolio.copy()
        
        #Check result on training
        strat.run_strategy(market_data=train_data,
                           portfolio = portfolio_train)
        
        train_result = strat.result
        
        #Check results on test
        portfolio_test = portfolio.copy()
        strat.run_strategy(market_data=test_data,
                           portfolio = portfolio_test)
        
        test_result =strat.result
        
        
        #Analyse results
        from ResultsAnalyser import ResultsAnalyser
        train_result = pd.merge(train_result,DJIA,how='inner',left_index=True, right_index=True)
        
        ra_train = ResultsAnalyser(data = train_result,
                                   valueIndex = 'Value',
                                   referenceIndex = 'DJIA' )
        
        rets = ra_train.get_returns(annualing_factor=1, useReference=True)
        rets += 1
        cumrets = rets.cumprod(axis=0)
#         print cumrets.tail()
        cumrets.plot()
        plt.show()
#       print train_result.head()
        print "Sharpe sa train", ra_train.sharpe_ratio()
        print "Sharpe reference train", ra_train.sharpe_ratio(useMarketRef=True)
        
        
        
        
        test_result = pd.merge(test_result,DJIA,how='inner',left_index=True, right_index=True)
        ra_test = ResultsAnalyser(data = test_result,
                                   valueIndex = 'Value',
                                   referenceIndex = 'DJIA' )
        
        rets = ra_test.get_returns(annualing_factor=1, useReference=True)
        rets += 1
        cumrets = rets.cumprod(axis=0)
#         print cumrets.tail()
        cumrets.plot()
        plt.show()
#       print train_result.head()
        print "Sharpe sa test", ra_test.sharpe_ratio()
        print "Sharpe reference test", ra_test.sharpe_ratio(useMarketRef=True)
        
       
        
#         #Coded no strat tester
#         ema = pd.ewma(train_data, ema_par)
#         momentum = ema.pct_change(lookback)
#         ranks = momentum.iloc[-lag].rank()
#         
#         #Pick top 10 and with momentum >0
#         port_series = ranks.iloc[0:10][ ranks.iloc[0:10]>0].index.values
# #         print port_series
#         
#         #Create weightings
#         weights = np.ones(len(port_series))
#         weights = np.linspace(1,len(port_series),len(port_series))
#         weights = weights/sum(weights)
# #       print sum(weights)
#         
#         portfolio = (weights*train_data[port_series]).sum(axis=1)
#         portfolio.plot()
#         DJIA.loc[train_periods[0][0]:train_periods[0][1]].plot()
# #         plt.show()
#          
#         portfolio_test = (weights*test_data[port_series]).sum(axis=1)
#         portfolio.plot()
#         DJIA.loc[test_periods[0][0]:test_periods[0][1]].plot()
# #         plt.show()
#          
#         
#         
#   
#             
        
        
    
    test_strat()
        
        
        
        
        