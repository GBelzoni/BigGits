import numpy
import scipy
import pandas
import matplotlib.pyplot as plt


# pandas plot example
ts = pandas.Series(numpy.random.randn(1000), index=pandas.DateRange('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
plt.show()

#Constructing rolling mean
pandas.rolling_mean(ts,60).plot()
ts.plot()
plt.show()

#Dataframe construction
df = pandas.DataFrame(ts)
print df
len(df)
df.index[1]
df.ix[1]

#Constructing empty DF and appending row - slow
df2 = pandas.DataFrame(columns = ( 'Time', 'Value', 'Signal'))
df= pandas.DataFrame({'a':[1],'b':[1], 'c':[1]})
##TODO
row = {'Time':2,'Value':2, 'Signal':2}
df2.append(row, ignore_index=True)

#Constructing DF from np array (mixed type) - faster
data = numpy.zeros(3,dtype=[('Time','f4'),('Value','f4'),('Signal','a10')])
print data
res_row = (1, 100,'stuff')
data[0]  = res_row   
print data
df = pandas.DataFrame(data)   
print df

#Read csv
data = pandas.read_csv("Documents/R/StrategyTester/Data/AORD.csv",parse_dates=True)
type(data)
isinstance(data, pandas.DataFrame)
list(data.columns)
type(data.index[1])

#Finding value at risk, etl
series = data['AORD.Close']
series.quantile(0.05)
diffs = series.diff()
#sort is in place
diffs.sort()
diffs.plot()
plt.show()
length = len(diffs)
#Calcing var as 5 quantile
vindex = numpy.floor(0.05*length)
vindex = int(vindex)
diffs.iloc[vindex] #var
diffs.quantile(0.05) #check
#etl
etl = numpy.average(diffs.iloc[0:vindex].values)


#initialising dat from zeroes
dat = -3*numpy.ones([10])
dat = pandas.DataFrame(dat)
dat.applymap( lambda x : max(x,0))
dat.abs()
dat**2
dat.mean(axis, skipna, level)

#Setting/Dropping index
import pandas as pd    
l1 = ['a','b','c','h','i']
l2 = ['a','b','c','d','e']
df = pd.DataFrame(zip(l1,l2))

df.set_index(keys=0,drop=False,inplace=True)
df.reset_index( drop=True)
