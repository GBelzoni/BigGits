'''
Created on Dec 8, 2012

@author: phcostello
'''
import unittest
import time
import strategy_tester.market_data as md
import strategy_tester.trade as td
import strategy_tester.Portfolio as pt
import strategy_tester.TradeStrategyClasses as tsc
import pickle

class Test(unittest.TestCase):

    def setUp(self):
        
        self.testdir = '/home/phcostello/Documents/workspace/FinancePython/strategy_tester/test/'
        self.SP500 =  pickle.load(open(self.testdir + 'SP500.pkl' ))

        
    def tearDown(self):
        pass


    def test_upd_signal(self):
        #Have manually gone through data AORD and found out where first buy is at
        self.assertTrue(self.trade_strat.upd_signal() == 'buy')

    def tes_upd_portfolio(self):
        
        #Test buy check that adds trade and that notional is updated
        
        #Test sell check that adds trade and that notional is updated
        pass
        
    
    def test_run_strategy(self):
        
        start_time = time.clock()
        self.trade_strat.run_strategy()
        print time.clock() - start_time, "seconds"
        print self.trade_strat.result
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()