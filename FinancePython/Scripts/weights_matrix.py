from BoilerPlate import *


df= pd.read_csv('Results/StrategyMSeries_Momentum/DJIA_weight_matrix.csv', index_col='Date')
df.index = pd.to_datetime(df.index)
df.info()
d1 = datetime.datetime(2013,4,1)
d2 = datetime.datetime(2013,6,28)

df.loc[d1:d2]
pd.Da
dfred = df.loc[d1:d2]
dfred = dfred.dropna(axis=1)
dfred.head()
dfred.to_csv('Results/StrategyMSeries_Momentum/DJIA_weight_matrix_red.csv')

