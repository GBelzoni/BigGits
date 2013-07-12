
import pandas as pd
import numpy as np
randn = np.random.randn

data = randn(2,4,5)
wp = pd.Panel(data)
#print data 
#print wp

df1 = pd.DataFrame( { 'a': [1,1], 'b': [2,2]})
df2 = pd.DataFrame( { 'a': [3,3], 'b': [4,4]})

#print df1
#print df2
##
#
#You have to make sure you have correct column names otherwise when you 
#append cols you get prob
dfdummy = pd.DataFrame( np.zeros([2,2]), columns=['a','b'])
pnl = pd.Panel({'dummy': dfdummy})
pnl['df1']=df1
pnl['df2']=df2
#print pnl['df2']

#Or make dict first

datadict = {}
datadict['df1']=df1
datadict['df2']=df2

pnl2 = pd.Panel(datadict)
print "pnl2['df1']" 
print pnl2['df1']



#pnl2 = pd.Panel({})
#
#