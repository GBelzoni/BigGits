'''
Created on Jul 3, 2013

@author: phco[stello
'''
import pandas as pd
from DataHandler.DBReader import DBReader
import os
import pickle
from strategy_tester.market_data import pairs_md
import matplotlib.pyplot as plt
from itertools import combinations
import statsmodels.api as sm

os.chdir('/home/phcostello/Documents/workspace/BigGits/FinancePython')

if __name__ == '__main__':
    
    
    
    def load_data_from_DB():

        con_string = "/home/phcostello/Documents/Data/FinanceData.sqlite"
    
        dbreader = DBReader( con_string )
        info = dbreader.seriesList
        dbinfo = info[info['Index'] == 'DJIA'] 
        dbinfo = dbinfo[dbinfo['Type']=='equity'] 
        series_names = dbinfo['SeriesName'].tolist()
        
        data_dict = { sn : dbreader.readSeries(sn) for sn in series_names}
        print 'number of series read', len(data_dict)
        
        panel_series = pd.Panel.from_dict(data_dict, 
                                          intersect=True, 
                                          orient = 'minor')
        print 'panel items', panel_series.items
#        panel_series.set_index['Date']
        dim = 'Adj_Close'
        df_allseries_dim = panel_series[dim]
        
        pickle.dump(df_allseries_dim,open('pickle_jar/DJIA_AdjClose.pkl','w'))
        
        print df_allseries_dim
    

    ######### Finding I1 Series ##################################################
    
    def CheckingWhichAreI1():
        FTSE100_AdjClose = pickle.load(open('pickle_jar/DJIA_AdjClose.pkl'))
    
        resultsI1 = []
        
        for nm in FTSE100_AdjClose.columns.tolist():
        
            seriesX = FTSE100_AdjClose[nm]
            seriesX = seriesX[seriesX.notnull()]
            adf_pval = sm.tsa.adfuller(seriesX)[1]
            
            resultsI1.append([nm, adf_pval])
            
            print 'Series {0} had Adf_pval {1}'.format(nm,adf_pval)
        
        
        return resultsI1
        
    
#    resultsI1 = CheckingWhichAreI1()
#    resultsI1 = pd.DataFrame( resultsI1, columns = ['SeriesName','Adf_val'])
#    resultsI1[ resultsI1['Adf_val'] < 0.05] 
#    pickle.dump(resultsI1 , open('pickle_jar/results_DJIA_I1.pkl','wb'))
    resultsI1 = pickle.load( open('pickle_jar/results_DJIA_I1.pkl','rb'))
    
    print len(resultsI1)
    
    I1seriesNames = resultsI1['SeriesName'].tolist()
    
    print I1seriesNames
    ######### Finding Cointegrated series #################################
    
    
    
    def search_for_CI(series_names):
        
        FTSE100_AdjClose = pickle.load(open('pickle_jar/DJIA_AdjClose.pkl'))
        results = []
        combs = [cm for cm in  combinations(series_names,2)]
        
        i = 0
        for cmb in combs:
            
            print 'Starting row', i
            i +=1
            
            seriesX_name = cmb[0]
            seriesY_name = cmb[1]
            
            try:
                seriesX = FTSE100_AdjClose[seriesX_name]
                seriesX = seriesX[seriesX.notnull()]
                
                seriesY = FTSE100_AdjClose[seriesY_name]
                seriesY = seriesY[seriesY.notnull()]
            
                df_coint_check = pd.merge(pd.DataFrame(seriesX),
                                      pd.DataFrame(seriesY), 
                                      how='inner', 
                                      left_index=True, 
                                      right_index=True)
                
                df_coint_check.columns = ['x','y']
                pmd = pairs_md(df_coint_check, xInd='x', yInd = 'y')
                pmd.fitOLS()
                reg_params = pmd.results.params.tolist()
                adf_pvalue = pmd.adfResids()[1]
                
                results.append([seriesX_name, seriesY_name, adf_pvalue] + reg_params)
                
            except ValueError as e:
                
                print "Problem with pairs X = {0}, Y = {1}".format(seriesX_name,seriesY_name)
                print "Error is ", e.what()
            
        
        pickle.dump(results , open('pickle_jar/results_DJIA_coint.pkl','wb'))
        
    
    ######### Analysing Results ###################################### 
    
#    load_data_from_DB()
#    search_for_CI(I1seriesNames)
    
    def plots():
        FTSE100_AdjClose = pickle.load(open('pickle_jar/DJIA_AdjClose.pkl'))
        results = pickle.load(open('pickle_jar/results_DJIA_coint.pkl','rb'))        
        results = pd.DataFrame(results, columns = ['SeriesX_name','SeriesY_name','Adf_pvalue', 'reg_scaling', 'reg_intercept'])
        results = results.sort('Adf_pvalue', ascending=True)
        
        print results.head(15)
        
        for row in range(0,5):
            rowinfo = results.iloc[row].tolist()
            seriesX = FTSE100_AdjClose[rowinfo[0]]
            seriesX = seriesX[seriesX.notnull()]
            
            seriesY = FTSE100_AdjClose[rowinfo[1]]
            seriesY = seriesY[seriesY.notnull()]
            
            
            df_coint_check = pd.merge(pd.DataFrame(seriesX),
                                  pd.DataFrame(seriesY), 
                                  how='inner', 
                                  left_index=True, 
                                  right_index=True)
            
            
            df_coint_check.columns = ['x','y']
            pmd = pairs_md(df_coint_check, xInd='x', yInd = 'y')
            pmd.fitOLS()
            pmd.generateTradeSigs(windowLength=20, entryScale=2, exitScale=0)
            
            pmd.plot_spreadAndSignals()
            plt.title( "Row {3}: Spread between {0} and {1}, scaling {2}".format(rowinfo[0],rowinfo[1],rowinfo[3], row))
            
            
            
        plt.show()

    plots()
    
#    
#    
#    
#    
#    
#    
#     