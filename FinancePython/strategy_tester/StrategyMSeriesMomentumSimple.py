'''
Created on Aug 7, 2013

@author: phcostello
'''

from BoilerPlate import *
import TradeStrategyClasses as tsc
from trade import TradeEquity
from Portfolio import Portfolio
from pandas.tseries.offsets import Day

class StrategyMSeriesMomentum(tsc.Trade_Strategy):
    '''
    This strategy scores series by momentum, the makes portfolio with given weights
    '''

    def __init__(self, 
                 rebalance_dates,
                 ema_par=100, 
                 lookback=10, 
                 lag=0, 
                 top_scores_num = 10):
        '''
        Constructor
        '''
        tsc.Trade_Strategy.__init__(self)
        self.ema_par=ema_par
        self.lookback = lookback
        self.lag = lag
        self.top_scores_num = top_scores_num
        self.rebalance_dates = rebalance_dates
        
        #This has top weights and rankings, plus series as index
#         self.weights_matrix = None
        
    
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
    
    def fit(self,data):
        '''
        Scores indexes momentum over period and calculates weighting vector
        stores results in internal parameters
        '''
        
        #set up ranges for training rebalancing
        periods = [ [self.rebalance_dates[i], self.rebalance_dates[i+1], self.rebalance_dates[i+2] ] for i in range(0, len(self.rebalance_dates)-2)]
        
        #setup result matrix
        
        
        self.weights_matrix = pd.DataFrame(np.zeros(data.shape))
        self.weights_matrix.columns = data.columns
        self.weights_matrix.index = data.index
        
        for prd in periods:
            
            this_data = data.loc[prd[0]:prd[1]]
            
            #Drop any series missing data for the period
#             this_data = this_data.dropna(axis=1)
            
            ema = pd.ewma(this_data, self.ema_par)
            momentum = ema.pct_change(self.lookback)
            ranks = momentum.iloc[-self.lag].rank(ascending=False).order()
            #Pick top X and with momentum >0
            topn = self.top_scores_num
            port_series = ranks.iloc[0:topn][ ranks.iloc[0:topn]>0].index
            #Create weightings
            #weights = np.ones(len(port_series))
            ranks_weights = pd.DataFrame((ranks[port_series]).copy(),columns =['Ranks'])
            
            #Weight top 10 by shares in ranking
            this_weights = (ranks_weights['Ranks'].max() - ranks_weights['Ranks']+1)/sum(ranks_weights['Ranks'])
            
            
            
            endTest = (prd[2]-Day())
            self.weights_matrix.loc[prd[1]:endTest] += this_weights 
    
        
    def run_strategy(self, data):
        '''
        Calcs returns
        '''
        
        #set up ranges for training rebalancing
        periods = [ [self.rebalance_dates[i], self.rebalance_dates[i+1] ] for i in range(0, len(self.rebalance_dates)-1)]
        
        portfolio = pd.DataFrame(np.zeros(data.shape[0],),index=data.index)
        
        for prd in periods[1:]:
            
            this_rets = data[prd[0]:prd[1]].pct_change()
            this_rets.iloc[0] = 0.0
            #Drop any series missing data for the period
#             this_rets.dropna(axis=1)
            
            this_rets.to_csv('Results/tmptmptmp.csv')
            ret_index = (1+this_rets).cumprod()
            
            ret_index.iloc[0] = 1
            this_portfolio = (ret_index*self.weights_matrix).sum(axis=1,skipna=True)
            this_port_rets = this_portfolio.pct_change()
            
            endDate = prd[1] - Day()
            common_index = portfolio.loc[prd[0]:endDate].index
            
            portfolio.loc[common_index]+=this_port_rets.loc[common_index]
            
        
        port_ret_index = (1+portfolio).cumprod()
        port_ret_index.iloc[0]=1
        self.result = pd.DataFrame(port_ret_index)
        self.result.columns = ['Value']
         
        
    
    
    
    
if __name__ == '__main__':
    
    def test_DJIA_strat(plots = False):
        
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
        
        
        #Do example
        
        #Setup strat
        
        #parameters
        top_scores_num = 5
        ema_par = 50
        lag = 1
        lookback = 10
        
        #Initialise strat
        strat = StrategyMSeriesMomentum(periods_dates,
                                        ema_par,
                                        lookback,
                                        lag,
                                        top_scores_num)
         
        #Train - ie pick top X momentum and weights
        #want generate weight matrix of constant weights between rebalance dates
        strat.fit(dataObj)
        print strat.rebalance_dates
        
        rebalance_flag = pd.DataFrame( [True]*len(strat.rebalance_dates) , index = strat.rebalance_dates)
        rebalance_flag = rebalance_flag.resample('B')
        rebalance_flag[pd.isnull(rebalance_flag)] = False
        rebalance_flag.index.name = 'date'
        rebalance_flag.columns = ['Rebalance_Flag']
        rebalance_flag.to_csv('Results/StrategyMSeries_Momentum/rebalance_flag.csv')
        
        
#         strat.weights_matrix.to_csv('Results/StrategyMSeries_Momentum/DJIA_weight_matrix.csv')
        sumweight =strat.weights_matrix.sum(axis=1,skipna=True)
#         sumweight.plot()
        strat.run_strategy(data=dataObj)
        res = strat.result
        
        DJIAindex = (DJIA/(DJIA.iloc[0]))
        
        tstart = datetime.datetime(2012,7,2)
        tend = datetime.datetime(2013,7,4)
        
        
        pltData = pd.merge(res,DJIAindex,how='inner',left_index=True,right_index=True)
        if plots:
            pltData.loc[tstart:tend].plot()
            plt.show()
             
        from ResultsAnalyser import ResultsAnalyser
        ra = ResultsAnalyser(pltData,valueIndex='Value',referenceIndex='DJIA')
        ra.set_results_range(tstart, tend)
        
        
         
        print "sa sharpe ratio", ra.sharpe_ratio()
        print "sa returns", ra.get_cumulative_return().iloc[-1]
        print "sa vol", ra.get_volatility(annualising_scalar=1,returns = True)
        print "to ref sharpe ratio", ra.sharpe_ratio(useMarketRef=True)

        balance_matrix = strat.weights_matrix.loc[strat.rebalance_dates]#.iloc[-5:-1]
        balance_matrix.to_csv('Results/StrategyMSeries_Momentum/balance_matrix.csv')
     
    def test_SP500_strat( ):
        
        #Already pickled DJIA_AdjClose
        dataObj = pickle.load(open('pickle_jar/SP500_AdjClose.pkl'))
        dataObj.to_csv('Results/tmptmp.csv')
        
        ####THIS IS WEIRD THAT DB HAS ADJ CLOSE AS VALUE
        ###NEED TO ADD TO PARSING IN READ DATA
        dataObj = dataObj.replace('Adj Close',np.nan).astype('float64')

        
        #Read DJIA index
        dbr = DBReader()
        ref_series = dbr.readSeries(seriesName = 'SP500')
        ref_series = ref_series['Adj_Close']
        ref_series = pd.DataFrame(ref_series)#,columns=['DJIA'])
        ref_series.columns = ['SP500']
        
        series_list = dataObj.columns.values
        
        #setup train/test ranges
        period_length_1 = 90
        period_length_2 = 180 
        start_date = dataObj.index[0]
        end_date = dataObj.index[-1]
#         start_date = datetime.datetime(2009,1,1)
#         end_date = datetime.datetime(2010,1,1)
        periods_dates = pd.date_range(start_date,end_date,freq = '3BMS')
        
        
        #Do example
        
        #Setup strat
        
        #parameters
        top_scores_num = 10
        ema_par = 100
        lag = 3
        lookback = 40
        
        #Initialise strat
        strat = StrategyMSeriesMomentum(periods_dates,
                                        ema_par,
                                        lookback,
                                        lag,
                                        top_scores_num)
         
        #Train - ie pick top X momentum and weights
        #want generate weight matrix of constant weights between rebalance dates
        strat.fit(dataObj)
        
        
        sumweight =strat.weights_matrix.sum(axis=1,skipna=True)
#         sumweight.plot()
        strat.run_strategy(data=dataObj)
        res = strat.result
        
        ref_seriesindex = (ref_series/(ref_series.iloc[0]))
        
        pltData = pd.merge(res,ref_seriesindex,how='inner',left_index=True,right_index=True)
        pltData.plot()
        plt.show()
        
        from ResultsAnalyser import ResultsAnalyser
        ra = ResultsAnalyser(pltData,valueIndex='Value',referenceIndex='SP500')
        tstart = datetime.datetime(2010,6,1)
        tend = datetime.datetime(2013,6,1)
        ra.set_results_range(tstart, tend)
        print "sa sharpe ratio", ra.sharpe_ratio()
        print "to ref sharpe ratio", ra.sharpe_ratio(useMarketRef=True)

            
    test_DJIA_strat(plots=True)
        
        
        
        
        