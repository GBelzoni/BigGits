from DataHandler.DBReader import DBReader
import pickle

dbr = DBReader()
SP500 = dbr.readSeries('SP500')
print SP500.info()
testdir = '/home/phcostello/Documents/workspace/FinancePython/strategy_tester/test/'
#pickle.dump(SP500,open( testdir + 'SP500.pkl','wb'))
SP5002 = pickle.load(open(testdir + 'SP500.pkl'))
SP5002.info()