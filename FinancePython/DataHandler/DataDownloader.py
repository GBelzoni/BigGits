'''
Created on Jun 13, 2013

@author: phcostello
'''

import Quandl
import pandas as pd
from pandas.io.data import DataReader
from datetime import datetime, timedelta, date
import pandas.io.sql as psql
import logging

from DataHandlerBase import DataHandlerBase

class DataDownloader(DataHandlerBase):
   
    '''This class handles updating the series tracked, and handles downloading data'''
    
    def updateRangeInfo(self,seriesNames, logfile = None):
        ''' updates the range info in seriesList so that it matches what's in db.
        should be run before/after importing new series
        '''

        
        errortables = []
        
        if logfile !=None:
            logging.basicConfig(filename= logfile, filemode='w', level = logging.ERROR)
       
        self.connect()
        
        for name in seriesNames:
            
            logging.info("updateRangeInfo for {}".format(name))
            sqlRead = "SELECT Date FROM {0}".format(name)
            
            #Read series data range
            try:
                dates = psql.read_frame(sqlRead, 
                                       con = self.con
                                       )
            except Exception as e:
                errortables.append(name)
                logging.error("updateRangeInfo: Reading table, encountered error <<{0}>>".format(e))
                continue
               
               
            #Convert to datetime objects
            dates = dates.apply(pd.to_datetime)
            StartRange = dates.min().iloc[0] #still series object so have to get data
            EndRange = dates.max().iloc[0]
            
            #Construct sql update query
            sqlWrite = "UPDATE SeriesList SET StartRange = '{0}', ".format(StartRange)
            sqlWrite += "EndRange = '{0}' ".format(EndRange)
            sqlWrite += "WHERE SeriesName = '{0}';".format(name)
            
            #print sqlWrite
            
            cur = self.con.cursor()
            
            try:
                cur.execute(sqlWrite)
            
            except Exception as e:
                logging.error("updateRangeInfo: Error executing write dates, encountered error <<{0}>>".format(e))
                errortables.append(name)
                continue
            
            else:     
                self.con.commit()
                
            
        self.disconnect()
        return errortables
        
    def updateSeriesData(self,seriesNames, maxDate = None, logfile = None):
        
        errortables = []
        
        
        #self.updateRangeInfo(seriesNames=seriesNames)
        
        if logfile !=None:
            logging.basicConfig(filename= logfile, filemode='w', level = logging.DEBUG)
        else:
            logging.basicConfig(level = logging.INFO)
        #Get info for series
        info = self.infoSeries(seriesNames)
        
        #Connect to db
        self.connect()
        
        
        
        for series in info.iterrows():
            
            #print "Starting loop for", series
            #Read relevant info
            series = series[1]
            endDate = pd.to_datetime(series['EndRange'])
            if isinstance(endDate, datetime)==False:
                continue
            endDate = endDate.date()
            
            seriesName = series['SeriesName']
            lookupTicker = series['LookupTicker']
            source = series['Source']
            
            #add logging message
            logging.info("updateData for {0}".format(seriesName))
            
            #Update Tables that have are less than maxDate
            
            if endDate >= maxDate :
                logging.info("Table {0} already up to date".format(seriesName))
            
            else:
                try:
                    print "trying readData"
                    endDate += timedelta(days=1)
                    data = self.readData(lookupTicker, source, endDate, maxDate)
                except Exception as e:
                    logging.error(e)
                    errortables.append(seriesName)
                    continue
                try:
                    self.writeFrameToDB(data, seriesName)
                except:
                    errortables.append(seriesName)
                    continue
                
        #self.updateRangeInfo(seriesNames=seriesNames)
        
        self.disconnect()
       
        return errortables
         
    def backdateSeriesData(self,seriesNames, minDate = None, logfile = None):
        
        errortables = []
        
        
        #self.updateRangeInfo(seriesNames=seriesNames)
        
        if logfile !=None:
            logging.basicConfig(filename= logfile, filemode='w', level = logging.DEBUG)
        else:
            logging.basicConfig(level = logging.INFO)
        #Get info for series
        info = self.infoSeries(seriesNames)
        
        #Connect to db
        self.connect()
        
        
        
        for series in info.iterrows():
            
            #print "Starting loop for", series
            #Read relevant info
            series = series[1]
            startDate = pd.to_datetime(series['StartRange'])
            if isinstance(startDate, datetime)==False:
                continue
            startDate = startDate.date()
            
            seriesName = series['SeriesName']
            lookupTicker = series['LookupTicker']
            source = series['Source']
            
            #add logging message
            logging.info("updateData for {0}".format(seriesName))
            
            #Update Tables that have are less than maxDate
            
            if startDate <= minDate :
                logging.info("Table {0} already up to date".format(seriesName))
            
            else:
                try:
                    print "trying readData"
                    startDate -= timedelta(days=1)
                    data = self.readData(lookupTicker, source, minDate, startDate)
                except Exception as e:
                    logging.error(e)
                    errortables.append(seriesName)
                    continue
                try:
                    self.writeFrameToDB(data, seriesName)
                except:
                    errortables.append(seriesName)
                    continue
                
        #self.updateRangeInfo(seriesNames=seriesNames)
        
        self.disconnect()
       
        return errortables
      
    def readData(self, lookupTicker, source, start, end):
        
        '''Read the data - assumes start and end are datetime.date objects'''
        
        try:  
            lookupTicker = str(lookupTicker)
            if source == 'Quandl':
                #use Quandl reader
                start = str(start)
                end = str(end)
                data = Quandl.get(lookupTicker,
                                  authtoken = self.quandlAuthToken,
                                  trim_start = start, 
                                  trim_end= end)
            else:
                #use pandas.io DataReader
                data = DataReader(lookupTicker, source , start, end)
                
            data = data.reset_index()
            logging.info("Read ticker {}".format(lookupTicker))
        except:
            logging.error("importData: Can't read ticker {}".format(lookupTicker))
            raise
        else:
            return data
        
    def writeFrameToDB(self, df, SeriesName):
        
        #Write to db
        try:
            self.connect()
            psql.write_frame( df, SeriesName, self.con, if_exists='append', safe_names=False)
            self.con.commit()
            logging.info("Wrote series ()".format(SeriesName))
        except:
            logging.error("Problems with {}".format(SeriesName))
            raise
        finally:
            self.disconnect()
        
    def addSeriesToUpdateList(self, filename, newType = False, newSource=False, newIndex = False):
        
        logging.basicConfig(level = logging.DEBUG)
        self.connect()
        
        #TODO maybe add these as data tables to update
        #validation lists
        existing_types = set(self.seriesList['Type'])
        existing_sources = set(self.seriesList['Source'])
        existing_series = set(self.seriesList['SeriesName'])
        existing_indexes = set(self.seriesList['Index'])
        
        
        ''' Adds series to be updated and checks info is ok'''
        thisSeriesList = pd.read_csv(filename)
        
        #Check correct colnames
        if not(self.seriesList.columns == list(thisSeriesList.columns)).all():
            print self.seriesList.columns
            print thisSeriesList.columns
            raise ValueError('Columns (names) in import file are incorrect')
        
        #Strip whitespace in table values
        thisSeriesList = thisSeriesList.applymap(lambda x : x.strip() )
        
        #Convert Start and End to Datetime
        thisSeriesList[['StartRange', 'EndRange']] = thisSeriesList[['StartRange', 'EndRange']].applymap(pd.to_datetime)
        
        #Append to SeriesList
        
        for row in thisSeriesList.iterrows():
            
            row = row[1] #itterow is tuple with second arg = value
            
            #Replace problem characters in name with blank
            row['SeriesName'] = row['SeriesName'].replace(".",'')
            row['SeriesName'] = row['SeriesName'].replace("&",'')
            row['SeriesName'] = row['SeriesName'].replace("-",'')
            row['SeriesName'] = row['SeriesName'].replace(",",'')
            row['SeriesName'] = row['SeriesName'].replace("'",'')
            
            
            #Replace whitespace by "_"
            row['SeriesName'] = row['SeriesName'].replace(" ",'_')
            
            
            
            #Check type is allowable
            if (row['Type'] not in existing_types) and not newType:
                logging.error('Series {0} has type {1} not in existing types'.format(row['SeriesName'],row['Type']))
                continue
            if (row['Index'] not in existing_indexes) and not newIndex:
                logging.error('Series {0} has index {1} not in existing indexes'.format(row['SeriesName'],row['Index']))
                continue
            if row['Source'] not in existing_sources and not newSource:
                logging.error('Series {0} has source {1} not in existing sources'.format(row['SeriesName'],row['Source']))
                continue
            if row['SeriesName'] in existing_series:
                logging.error('Series {0} is already in existing SeriesNames'.format(row['SeriesName']))
                continue
                
            
            
            #if passes all checks then write to db
            logging.info('Wrote {} to SeriesList'.format(row['SeriesName']))
            row = pd.DataFrame(row).transpose()
            psql.write_frame(row, 'SeriesList', self.con, if_exists='append')
            self.con.commit()
                
        
    
  
if __name__ == "__main__":

    
    conString = "/home/phcostello/Documents/Data/FinanceData.sqlite"
    dd = DataDownloader(conString)
 
    #Update SeriesList with data from csv. Does some checking and formatting    
    #TODO: Need to make this so it doesn't double up series
    #TODO: Need to add tables of sources and types

    def addSeriesToUL(newIndex=False):
        filename = "../SeriesInfoCSV/DowJonesIndustrial.csv"
        dd.addSeriesToUpdateList(filename, newIndex=newIndex)
    
    def updateDataSeries(updateData = False):
        #Get names of series to update
        allinfo = dd.seriesList
        IndexFilter = 'DJIA'
        IndexInfo = allinfo[allinfo['Index']==IndexFilter]
        print (IndexInfo['SeriesName'].unique())
        
        #names = IndexInfo['SeriesName'].unique()
        names = ['DJIA']
        print names
        
        
        #Update series to end
        end = (datetime.now().date()) #Has to be datetime.date object
        if updateData:
            errortables = dd.updateSeriesData(names, end) #This updates series in db
            print errortables
        dd.updateRangeInfo(names) #this updates info in tables to reflect series
      
    def backdateDataSeries(updateData = False):
        #Get names of series to update
        allinfo = dd.seriesList
        IndexFilter = 'SP500'
        IndexInfo = allinfo[allinfo['Index']==IndexFilter]
        print (IndexInfo['SeriesName'].unique())
        
        #names = IndexInfo['SeriesName'].unique()
        names = (IndexInfo['SeriesName'].unique())
        print names
        
        
        #Update series to end
        start = datetime(2005,1,1).date() #Has to be datetime.date object
        print start
        if updateData:
            errortables = dd.backdateSeriesData(names, start) #This updates series in db
            print errortables
            dd.updateRangeInfo(names) #this updates info in tables to reflect series
        
          
      
#    addSeriesToUL(newIndex=True)
#     updateDataSeries(updateData=False)
    backdateDataSeries(updateData=True)

### LITTTLE SCRIPT BELOW TO CLEAN OUT DB

#import sqlite3
#import pandas as pd
#import pandas.io.sql as psql

#Get all tables in db
#
#con = sqlite3.connect("/home/phcostello/Documents/Data/FinanceData.sqlite")    
#cur = con.cursor()
#
#sqlAllTable = "SELECT tbl_name FROM sqlite_master WHERE type = 'table'"
#dbtables = cur.execute(sqlAllTable)
#
#tblist = dbtables.fetchall()
#tblist = [ nm[0] for nm in tblist]
#len(set(tblist))
#
#
#sqlgetSeriesListNames = "SELECT SeriesName FROM SeriesList"
#seriesListNames = cur.execute(sqlgetSeriesListNames)
#slnames = seriesListNames.fetchall()
#slnames = [nm[0] for nm in slnames]
#
#slnames.append( 'SeriesList')
#slnames.index('SeriesList')
#len(slnames)
#
#len(set(slnames))
#rmFromDB = list(set(tblist).difference(set(slnames)))
#
#for tb in rmFromDB:
#    
#    cur.execute("DROP TABLE {} ".format(tb))
#    cur.commit()
    



        