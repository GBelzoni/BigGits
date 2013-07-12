'''
Created on Jun 23, 2013

@author: phcostello
'''

import pandas as pd
import numpy as np

class ResultsAnalyser(object):
    '''
    This class analyses fitted trade strategy object
    '''

    def __init__(self, strategy, referenceIndex = None):
        '''
        Takes a pandas dataframe object with a 'Date' index which should be of type date or datetime
        'Value' column which is timeseries of portfolio value
        'Signal' which is the trade signal
        '''
        self.result = strategy.result
        if referenceIndex != None:
            self.refIndex = strategy.market_data.core_data[referenceIndex]
        else:
            self.refIndex = None
            
        self.result_date_range = [ strategy.result.index[0], strategy.result.index[-1]]
        self.result_subset = self.result.loc[self.result_date_range[0]:self.result_date_range[-1]].copy()
         

    def set_results_range(self, tstart, tend, make_positive=False):
        
        '''
        Sets the range for which the results will be calculated for given run strategy
        make_positive adds min over range so returns, etc make sense
        '''
        self.result_date_range = [ tstart, tend]
        self.result_subset = self.result.loc[self.result_date_range[0]:self.result_date_range[-1]].copy()
        
        if make_positive:
            if min(self.result_subset['Value']) <= 0:
                #make the series >= 100 so that returns calc ok
                self.result_subset['Value'] += np.abs(min(self.result_subset['Value'])) + 100
            
    
    def reset_range_to_dafault(self): 
    
        self.result_date_range = [ self.result.index[0], self.result.index[-1]]
    
    def get_result(self):
        
        return self.result_subset.loc[self.result_date_range[0]:self.result_date_range[1]]
    
    def get_returns(self,cumperiods =1,useReference=False):
        
        ''' Calcs returns vs above referenceIndex, if None type then usual returns '''
        
        value = self.result_subset['Value']
        #data = self.result['Value'][self.result['Value'].notnull()] #Horrible line, but is just filtering out notnull values
        retsPort = pd.DataFrame(value.pct_change())
        #retsPort = (1+retsPort)**cumperiods-1 #For annualising
        
        if useReference == True:
            if self.refIndex == None:
                raise ValueError('No reference index set')
            
            retsRef = self.refIndex.pct_change()
            rets = pd.merge(retsPort,retsRef, how='inner',left_index=True,right_index=True)
            
        else:
            
            rets = retsPort
            rets['Reference'] = 0
            
        rets.columns = ['Portfolio','Reference']
        
        return rets
    
    def get_cumulative_return(self,cumperiods=1,useReference = False):
        
        rets = self.get_returns(1, useReference) #get 1period returns
        cumrets = (1+rets).cumprod()-1 #cumulate
        cumrets = (cumrets-1)**cumperiods + 1 #Apply annualising factor
        return cumrets
        
    def get_volatility(self, annualising_scalar = 1, returns = False):    
        
        '''Get volatility of portfolio value or returns'''
        if not(returns):
            value = self.result_subset['Value']
            return value.std()
        else:
            rets = self.get_returns(annualising_scalar, False) #get 1period returns
            
            return rets['Portfolio'].std()
        
        
    def sharpe_ratio(self, useMarketRef=False):
        
        ''' Calcs sharpe ratio vs marketRef, if None then riskfree rate assumed to be 0'''
        
        rets = self.get_returns(useReference=useMarketRef)
        
        if useMarketRef:
            retsOverRef = rets['Portfolio'] - rets['Reference']
        else:
            retsOverRef = rets['Portfolio']
        
        sr = retsOverRef.mean(skipna=True)/retsOverRef.std(skipna = True)
        return sr
    
    def sortino_ratio(self, useMarketRef= False, benchmarkRate = None ):
        
        ''' Calcs sortino ratio vs benchmark
        if False useMarketRef then refrate assumed to be be 0
        if no benchmark is None the benchmark target is zero
        '''
        
        rets = self.get_returns(useReference=useMarketRef)
        retsOverRef = rets['Portfolio'] - rets['Reference']
        
        if benchmarkRate == None:
            benchmarkRate = 0.0
            
        benchmarkSeries = benchmarkRate*np.zeros(len(retsOverRef))
        benchmarkSeries = pd.DataFrame(benchmarkSeries, index = retsOverRef.index)
        retsOverBenchMark = retsOverRef - benchmarkSeries
        
        #Calc numerator in sortino ratio
        numerator = retsOverBenchMark.mean()
        
        #Calc denominator in sortino, ie std for only returns over benchmark
        denominator = (retsOverBenchMark.abs() + retsOverBenchMark)/2.0 #Gives max(value,0)
        denominator = denominator**2 #Square values
        denominator = denominator.mean() #get mean
        denominator = np.sqrt(denominator)
        
        sortinor = numerator/denominator
        
        return sortinor[0]
    
    def draw_downs(self, percent = False):
        
        '''Calcs timeseries of current percentage drawdown'''
        
        #Calculate percentag Drawdowns 
        value = self.result_subset['Value']
        startt= value.index[0]
        Max = value[startt]
        dd = 0
        startDd = startt
        endDd = startt
        lengthDd = 0
        
        result = [[startt,Max,dd, startDd, endDd, lengthDd]]
        
        for t in value.index:
            
            Max = max(Max,value[t])
            if Max == value[t]:
                startDd = t
                dd = 0
            
            if percent:
                thisDd = (Max - value[t])/Max
            else:
                thisDd = (Max - value[t])
                    
            dd = max(dd, thisDd )
            if not(Max == value[t]):
                endDd = t
            
            lengthDd = startDd - endDd
            
            thisResult = [t,Max,dd, startDd, endDd, lengthDd]
            result.append(thisResult) 
        
        #Format results to dataframe
        columns = ['Time','MaxVal','Drawdown','StartDD','EndDD','LengthDD']
        result = pd.DataFrame(data=result, columns=columns)   
        result.set_index('Time', inplace=True)
        
        return result

    def max_draw_down_magnitude(self, percent = False):
        
        '''Calcs max drawdown'''
        
        dd = self.draw_downs(percent)
            
        maxDD = max(dd['Drawdown'])
        maxDDr = dd[dd['Drawdown'] == maxDD]
        
        return maxDDr.iloc[0]
    
    def max_draw_down_timelength(self, percent = False):
        
        '''Calcs max drawdown'''
        
        dd = self.draw_downs(percent)
        
        thisDD = dd.iloc[2:] #get rid of non types at start of series
        maxDD = min(thisDD['LengthDD'])
        maxDDr = thisDD[thisDD['LengthDD'] == maxDD]
        return maxDDr.transpose()
 
    def var(self, percentile = 0.05):
        '''Value at Risk'''
        #subset on range and then diff
        diffs = value = self.result_subset['Value'].diff()
        return diffs.quantile(percentile)
        
 
    def etl(self, percentile = 0.05):
        '''Expected Tail Loss'''
        
        
        diffs = value = self.result_subset['Value'].diff()
        
        #sort is in place
        diffs.sort()
        
        #Calcing var as as quantile of percentile
        length = len(diffs)
        vindex = np.floor(percentile*length)
        vindex = int(vindex)
        
        #etl
        etl = np.average(diffs.iloc[0:vindex].values)
        
        return etl
        
    
    def summary(self):
        
        '''Calcs summary of Sharpe and max drawdown'''
        
        print "Sharpe Ratio", self.sharpe_ratio()
        
        print "Value At Risk 0.05, 0.10"
        print self.var(0.05), self.var(0.05)
        print "Expected Tail Loss 0.05, 0.10"
        print self.etl(0.05), self.etl(0.05)
        print "MaxDrawDown (level)"
        print self.max_draw_down_magnitude()
        print ""
        print "MaxDrawDown (percent)"
        print self.max_draw_down_magnitude(percent = True)
        print ""
        print "MaxDrawDown time (level)"
        print self.max_draw_down_timelength()
        print ""
        print "MaxDrawDown time (percent)"
        print self.max_draw_down_timelength(percent = True)

class PairTradeAnalyser(ResultsAnalyser):
    '''
    This class analyses fitted trade strategy object
    '''
    
    def __init__(self, strategy, referenceIndex):
        '''
        Constructor
        '''
        ResultsAnalyser.__init__(self,strategy, referenceIndex)
        self.yscaling = strategy.market_data.results.params[0]
        self.adfPvalue= strategy.market_data.adfResids()[1]
    
    def summary(self):
            print "Summary for Pair Trade Strategy"
            print "Scaling", self.yscaling
            print "Adf p-value", self.adfPvalue
            print "Sharpe Ratio", self.sharpe_ratio()
            print "Value At Risk 0.05, 0.10"
            print self.var(0.05), self.var(0.10)
            print "Expected Tail Loss 0.05, 0.10"
            print self.etl(0.05), self.etl(0.10)
            print ""
            print "MaxDrawDown (level)"
            print self.max_draw_down_magnitude()
            print ""
            print "MaxDrawDown (percent)"
            print self.max_draw_down_magnitude(percent = True)
            print ""
            print "MaxDrawDown time (level)"
            print self.max_draw_down_timelength().loc['LengthDD']
            print ""
            print "MaxDrawDown time (percent)"
            print self.max_draw_down_timelength(percent = True).loc['LengthDD']
        
        
        