from BoilerPlate import *


##################Make par Table##########################
#Parameter table rows are
#[ pairs_labes, strategy_params, train_ranges, test_ranges ]
#
##series
#series_names = [['x', 'y']]
#
##strat pars
#entry_scale = np.arange(1,3.5,0.5)
#exit_scale = np.array([0,3.0,0.5])
#strat_parameters = [ [sw, lw] for sw in entry_scale for lw in exit_scale if sw < lw]
#
##Train - test date pars
#dmin = datetime.date(2011,1,1)
#num_periods = 3
#period_delta = datetime.timedelta(182)
##Create range start/end offsets from min data
#train_test_ranges = [ [dmin + i*period_delta, #train start
#                     dmin + (i+1)* period_delta, #train end
#                     dmin + (i+1)* period_delta, #test start
#                     dmin + (i+2)* period_delta] #test end
#                     for i in range(0,num_periods)]
#train_test_ranges
#
#strat_parameters[0] + train_test_ranges[0]
#
##Make table of parameters + dates
#parameter_table = [ [series + pars + date_ranges] for series in series_names for pars in strat_parameters for date_ranges in train_test_ranges]
#parameter_table


dbpath = "/home/phcostello/Documents/Data/FinanceData.sqlite"
dbreader = DBReader(dbpath)
SP500 = dbreader.readSeries("SP500")
BA = dbreader.readSeries("BA")
dim = 'Adj_Close'
SP500AdCl = SP500[dim]
BAAdCl = BA[dim]
dataObj = pd.merge(pd.DataFrame(BAAdCl), pd.DataFrame(SP500AdCl), how='inner',left_index = True, right_index = True)
dataObj.columns = ['y','x']

time0 = datetime.date(2012,1,1)
time1 = datetime.date(2012,1,10)
pd.Timestamp(time)
print dataObj.loc[time0:time1]

dataObj.index[-1]

