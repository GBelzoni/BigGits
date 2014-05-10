'''
Created on Jun 19, 2013

@author: phcostello
'''
from DataHandlerBase import DataHandlerBase
import pandas.io.sql as psql
import pandas as pd
import pickle
from BoilerPlate import *

class DBReader(DataHandlerBase):
    '''
    Simple class to read series from db
    '''

    def readSeries(self, seriesName):
        ''' Just reads data from db ''' 

        self.connect()
        sql = "SELECT * FROM {}".format(seriesName)
        try:
            data = psql.read_frame(sql, self.con)
        except Exception as e:
            print "Problem reading {0}, error is {1}".format(seriesName,e)
            raise e
        
        try:
            data = data.set_index('Date') #Set date index
        except Exception as e:
            print "Problem setting date index in series {0}, error is {1}".format(seriesName,e)
            raise e   
        
        try:
            data.index = pd.to_datetime(data.index)
            data.sort_index(inplace=True)
            return pd.DataFrame(data)
        except Exception as e:
            print "Problem converting date index using pd.to_datetime in series {0}, error is {1}".format(seriesName,e)   
            raise e
    
      
if __name__ == '__main__':
    
    def one_series( plot = False):
        import matplotlib.pyplot as plt
        dr = DBReader()
        name = 'SP500'
        df = dr.readSeries(name)
        print df.tail()
        if plot:
            df['Adj_Close'].plot()
            plt.show()
            
    def load_data_from_DB(pickle = False):

        con_string = "/home/phcostello/Documents/Data/FinanceData.sqlite"
    
        dbreader = DBReader( con_string )
        info = dbreader.seriesList
        dbinfo = info[info['Index'] == 'SP500'] 
        dbinfo = dbinfo[dbinfo['Type']=='equity'] 
        series_names = dbinfo['SeriesName'].tolist()
        
        data_dict = {}
        
        for sn in series_names:
            
            try:
                this_data =  dbreader.readSeries(sn)
                data_dict[sn] = this_data 
            except Exception as e:
                print 'Error', e
        
        
        print 'number of series read', len(data_dict)
        
        panel_series = pd.Panel.from_dict(data_dict, 
                                          intersect=True, 
                                          orient = 'minor')
        
        print 'panel items', panel_series.items

        dim = 'Adj_Close'
        df_allseries_dim = panel_series[dim]
        
        print df_allseries_dim
        
        if pickle:
            pickle.dump(df_allseries_dim,open('pickle_jar/SP500_AdjClose.pkl','w'))
        
        
    load_data_from_DB()
    