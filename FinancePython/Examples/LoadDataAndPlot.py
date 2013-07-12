'''
Created on Jun 26, 2013

@author: phcostello
'''

import DataHandler.DBReader as dbr
import pandas as pd
import strategy_tester.market_data as md
import matplotlib.pyplot as plt
from datetime import datetime
conString = "/home/phcostello/Documents/Data/FinanceData.sqlite"

if __name__ == '__main__':
    
    
    #Load data from db
    dr = dbr.DBReader(conString)
    dataFull = dr.readSeries('AAPL')
    print dataFull.head()
    plot = dataFull['Adj_Close'].plot()
    
    #Change Date Range
    d1 = datetime(2012,1,1).date()
    d2 = datetime(2012,6,1).date()
    
    data = dataFull.loc[d1:d2]
    print data.head()
    
    #Generate trade sigs for simple moving average
    sma = md.simple_ma_md(data)
    sma.generateTradeSig('AAPL',
                         'Adj_Close',
                         short_win = 10,
                         long_win = 20)
    sma.plot()
    
    #Generate trade sigs for exponentially weighted moving ave 
    ema = md.exponential_ma_md(data)
    ema.generateTradeSig('AAPL',
                         'Adj_Close',
                         short_span = 10,
                         long_span = 20)
    ema.plot()
    
    
    #Do something with pairs trades. Should add something about data downloading panels
    plt.show()
    
    
    
    