'''
Created on Dec 8, 2012

@author: phcostello
'''
import unittest
import numpy as np
import pandas as pd
import strategy_tester.market_data as md
import strategy_tester.trade as td
import pickle
import math


class Test(unittest.TestCase):


    def setUp(self):
        #setup market data#
        #First setup data as list
        data = np.array([1,0.05,100,0.1])
        data.resize(1,4)
        
        #Then converst to types needed for pricing
        columns = ["zero",
                   "rate",
                   "underlying",
                   "vol"]
        
        data = pd.DataFrame(data, columns = columns)
        self.md_for_option = md.market_data(data)

        self.testdir = '/home/phcostello/Documents/workspace/FinancePython/strategy_tester/test/'
        self.SP500 =  pickle.load(open(self.testdir + 'SP500.pkl' ))
        self.md_SP500 = md.market_data(self.SP500) 

    def tearDown(self):
        pass


    def test_cash(self):
        
        #setup cash trades to test
        Cash_Trade = td.TradeCash("TestCash",
                                  notional=100,
                                  rate_label='rate')
        
        #Do tests
        self.assertEqual(Cash_Trade.price(),1)
        self.assertEqual(Cash_Trade.value(),100)
        
        Cash_Trade.inflate(self.md_for_option,
                           period_length=1)
        
        rate = self.md_for_option['rate']
        print rate
        self.assertAlmostEqual(Cash_Trade.notional, math.exp(rate)*100)
        

    def test_equity(self):
        
        md = self.md_SP500
        time = self.md_SP500.index.values[0]
        #setup cash trades to test
        Equity_Trade = td.TradeEquity("TestEquity", 
                                      notional=100,
                                      price_series_label='Adj_Close')
        self.assertEqual( Equity_Trade.price(md,time),1132.99)
        self.assertEqual( Equity_Trade.value(md,time),113299.0,1)
        self.assertEqual( Equity_Trade.notional,100)
        
        
    def test_vanilla_euro_call(self):
        
        md = self.md_for_option
        time = 0
        #setup trades
        # Strik = 105
        # Underlying = 100
        # 12m expiry
        # vol = 10%
        # dividend =0 
        # risk free = 0.05
        # val = $4.046
        
        Vanilla_Euro_call = td.TradeVanillaEuroCall("TestEuroCall",
                                                     notional=100, 
                                                     strike=105, 
                                                     expiry=1)
        
        self.assertAlmostEqual(Vanilla_Euro_call.price(md, time),4.046,3)
        self.assertAlmostEqual(Vanilla_Euro_call.value(md, time),404.6097,3)
        self.assertAlmostEqual(Vanilla_Euro_call.delta(md, time),52.4758,3)
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()