'''
Created on Dec 7, 2012

@author: phcostello
'''
import unittest
from strategy_tester.market_data import market_data, simple_ma_md, pairs_md
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
zoo = importr('zoo')
import pandas as pd 
import pickle
import matplotlib.pyplot as plt

class MarketDataTest(unittest.TestCase):

    def setUp(self):
        #unittest.TestCase.setUp(self)
         
        #Setup
        self.testdir = '/home/phcostello/Documents/workspace/FinancePython/strategy_tester/test/'
        self.SP500 =  pickle.load(open(self.testdir + 'SP500.pkl' ))
#        self.zooData = zoo.zoo(self.AORD)        
#        self.AORD = ro.r('read.table("~/Documents/R/StrategyTester/Data/AORD.csv",header=T, sep=",")')
        
        
    def tearDown(self):
        unittest.TestCase.tearDown(self)    
     
    def test_addTechnical(self):
        '''Test that adding simple technical indicators'''
        
        md = market_data(self.SP500)

        #Test simple moving average
        sma = pd.rolling_mean(self.SP500['Adj_Close'], 10)
        md.addSMA('Adj_Close','sma',10)
        self.assertTrue((sma[10:] == md['sma'][10:]).all(), 'Add simple ma wrong')

        #Test adding exponetially weighted moving average works
        ema = pd.ewma(self.SP500['Adj_Close'], 10)
        md.addEMA('Adj_Close','ema', win_length=10)
        self.assertTrue((ema[10:] == md['ema'][10:]).all())

        #Test adding rolling stdev average works
        stdev = pd.rolling_std(self.SP500['Adj_Close'], 10)
        md.addSDev('Adj_Close','stdev', win_length=10)
        self.assertTrue((stdev[11:] == md['stdev'][11:]).all())

    def test_MD_simpleMovingAverage(self):
        
        #Test the moving averages are contructed with length >= 1
        sma = simple_ma_md(self.SP500)
        sma.generateTradeSig(seriesName = 'SP500', 
                             sig_index = 'Adj_Close', 
                             short_win = 10, 
                             long_win = 20)
        
        rm = pd.rolling_mean(sma['Adj_Close'],20)
        self.assertTrue((sma['MAl'][20:] == rm[20:]).all(), 'simpleMAshort not working')
        
        
    def test_pairs_md(self):
    #prepare data
        import DataHandler.DBReader as dbr
        dbpath = "/home/phcostello/Documents/Data/FinanceData.sqlite"
        dbreader = dbr.DBReader(dbpath)
        SP500 = dbreader.readSeries("SP500")
        BA = dbreader.readSeries("BA")
        dim = 'Adj_Close'
        SP500AdCl = SP500[dim]
        BAAdCl = BA[dim]
        dataObj = pd.merge(pd.DataFrame(BAAdCl), pd.DataFrame(SP500AdCl), how='inner',left_index = True, right_index = True)
        dataObj.columns = ['y','x']
        
        pmd = pairs_md(dataOb=dataObj,xInd='x',yInd='y')
        pmd.fitOLS()
        resid = pmd.results.resid
        resid.plot()
        rllstd = pd.rolling_std(resid,50)
        rllstd.plot()
        
        pmd.generateTradeSigs(50, entryScale=1, exitScale=0)
        self.assertTrue( (pmd['entryUpper'][50:] == rllstd[50:]).all())
        
        #Displays
#        pmd.printSummary()
#        print pmd.adfResids()
#        pmd.plot_spreadAndSignals()   
#        plt.show()
#        
        
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()