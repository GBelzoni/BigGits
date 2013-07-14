'''
Created on Jul 12, 2013

@author: phcostello
'''
from BoilerPlate import *

from DataHandler.DBReader import DBReader
from strategy_tester.ResultsAnalyser import ResultsAnalyser

if __name__ == '__main__':
    
    
    dbreader = DBReader()
    AAPL = dbreader.readSeries('AAPL')
    SP500 = dbreader.readSeries('SP500')
    print AAPL.info()
    print SP500.info()
    data = pd.merge(pd.DataFrame(AAPL['Adj_Close']),pd.DataFrame(SP500['Adj_Close']), how='inner', left_index=True, right_index=True)
    data_dict = {'AAPL':AAPL , 'SP500':SP500}
    panel_series = pd.Panel.from_dict(data_dict, 
                                          intersect=True, 
                                          orient = 'minor')
    print 'panel items', panel_series.items
    dim = 'Adj_Close'
    df_allseries_dim = panel_series[dim]
    
    
    dataRed = df_allseries_dim.loc['2011-1-1':'2012-1-1']
    
    rets = dataRed['SP500'].pct_change()
    
    print 'rets std', rets.std()
    print rets.head()
    
    #dataRed['SP500'].plot()
    plt.show()
    
    ra = ResultsAnalyser(dataRed,valueIndex = 'SP500')#, referenceIndex='SP500')
    print "vol", ra.get_volatility(annualising_scalar=1, returns=True)
    
    sharpe = ra.sharpe_ratio(useMarketRef=False)
    print sharpe
    