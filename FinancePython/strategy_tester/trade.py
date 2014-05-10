'''
Created on Nov 29, 2012

@author: phcostello
'''

import math
from scipy.stats import norm



class Trade(object):
    '''
    Trade Base object
    '''

    def __init__(self, name, type, notional):
        '''
        Constructor
        '''
        self.name = name
        self.type = type
        self.notional = notional
        
    def price(self, market_data, time):
        
        return "No price function implemented"
                
    def value(self, market_data, time):
        
        return self.price(market_data,time) * self.notional
    
class TradeCash(Trade):
    ### Basic risk free cash trade ###
    def __init__(self,name, notional,  rate_label = "rate"):
        Trade.__init__(self, name = name, type="Cash", notional=notional)           
        self.rate_label = rate_label
            
    def price(self,market_data,time):
        ### Assume nominal units so price = 1. ##
        
        return 1
    
    def value(self):
        ### Assume nominal units so price = 1. ##
        return self.notional
    
    def inflate(self,market_data,period_length):
            
        rate = market_data[self.rate_label]
        self.notional *= math.exp( rate * period_length)
        
    
    
class TradeEquity(Trade):
    
    ### simple equity ts ###
    
    def __init__(self, name, notional, price_series_label = 0):
        '''
        Can name the trade whatever you want
        price_series_label needs to match the price series in corresponding md object
        '''
        
        Trade.__init__(self, name = name, type="Equity", notional=notional)
        self.price_series_label = price_series_label
    
    def price(self, market_data, time):
        
        if self.price_series_label not in market_data.columns:
            raise ValueError('Equity price series label {} not in market data'.format(self.price_series_label))
        
        return market_data.at[time, self.price_series_label]
    
    def delta(self, market_data_slice):    
        
        return 1  

class TradeVanillaEuroCall(Trade):
    
    ### simple equity ts ###
    
    def __init__(self, name, notional, strike, expiry):
        
        Trade.__init__(self, name = name, type="VanillaEuroCall", notional=notional)
        self.strike = strike
        self.expiry = expiry
        
    def price(self, market_data, time):
        
        ### wrapper for bs analytic price ###
        
        r = market_data['rate']
        vol = market_data['vol']
        S = market_data['underlying']
        t0 = time
        K = self.strike
        expiry = self.expiry
        dividend = 0
        
        price = faf.BS_call(S, K, t0, expiry, r, dividend, vol)
        
        return price.ix[0]
        
    def delta(self, market_data, time):    
        
        #Calcs delta of trade. This is delta_option_price * notional_option
        
        r = market_data['rate']
        vol = market_data['vol']
        S = market_data['underlying']
        t0 = time
        K = self.strike
        expiry = self.expiry
        dividend = 0
        
        epsilon = 0.0001
        
        pricepldel = faf.BS_call(S+epsilon,K,t0,expiry,r,dividend,vol)
        pricemindel = faf.BS_call(S-epsilon,K,t0,expiry,r,dividend,vol)
    
        delta_price = (pricepldel - pricemindel)/(2* epsilon)
        delta = delta_price * self.notional
        
        return delta.ix[0]
    
if __name__ == '__main__':
   
    import numpy as np
    import pandas as pd
    
    r = 0.05
    dividend = 0.0
    vol = 0.1
    S = 100.0
    t0 = 0.0
    expiry = 0.001
    K = 100.0
    
     
    #setup market data#
    #First setup data as list
    data = np.array([r,S,vol])
    data.resize(1,3)
    
    #Then convert to types needed for pricing
    columns = ["rate",
               "underlying",
               "vol"]
    
    
    data = pd.DataFrame(data, columns = columns)
    md_cash = market_data(data)

    
    #Setup vanilla option trade
    tradeCall = TradeVanillaEuroCall(name = "Call",
                                        notional = 1,
                                        strike = 100,
                                        expiry = 0.5)
                                        
    price = tradeCall.price(md_cash,0)
    print "price = ", price
    delta = tradeCall.delta(md_cash,0)   
    print "delta = ", delta      
          
        