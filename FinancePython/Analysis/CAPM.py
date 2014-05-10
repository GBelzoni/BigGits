from BoilerPlate import *
from strategy_tester.ResultsAnalyser import ResultsAnalyser
from numpy.linalg import *
#Get DJIA data
#Already pickled DJIA_AdjClose
dataObj = pickle.load(open('pickle_jar/DJIA_AdjClose.pkl'))
#Read DJIA index
dbr = DBReader()
DJIA = dbr.readSeries( seriesName = 'DJIA')
DJIA = pd.DataFrame(DJIA)#,columns=['DJIA'])
DJIA.columns = ['DJIA']

SP500 = dbr.readSeries( seriesName = 'SP500')
SP500 = pd.DataFrame(SP500)
Benchmarks = pd.merge(DJIA,SP500,right_index=True,left_index=True)
Benchmarks = Benchmarks[['DJIA','Adj_Close']]
Benchmarks.columns = [['DJIA','SP500']]

dataObj.head()
series = dataObj.loc['2011':]

series.plot()
plt.show()
#Calc max sharpe from series
rets = series.pct_change()
from datetime import datetime
train_range = [datetime(2012,1,1),datetime(2013,1,1)]
def sharpe(rets):
    #return simple sharpe ratio
    return rets.mean()/rets.std()


rets_train = rets.loc[train_range[0]:train_range[1]]
rets_test = rets.loc[train_range[1]:]

# %cpaste

def CAPM(rets,rf=0,retMat = False, an_f = 252):
    '''
    fits CAPM to returns series
    Uses mean of rets and cov/cor matrix over ret period to fit
    Therefore this gives optimal choice of weights and max_sr over the 
    period
    
    Need to make version where feed in expected returns and cov matrix
    to see how it compares to optimal
    '''
    mn = rets.mean()
    sd = rets.std()
    sr = sharpe(rets)
    
    V = rets.cov()
    R = rets.corr()
    
    invV = inv(V)
    invR = inv(R.as_matrix())
    
    sr_vec = sr.values
    
    max_sr = np.sqrt(an_f*sr.dot(invR.dot(sr))) 
    weights = invV.dot(mn)
    weights /= sum(weights)
    weights = pd.DataFrame(weights, index = rets.columns)
    weights.columns = ['weights']
    
    if retMat:
        return max_sr, weights, V, R
    else:
        return max_sr, weights
        

max_sr_train_full, theo_weights_full = CAPM(rets_train)
max_sr_train_full
#Try picking top X by abs weights
sorted_weights = np.abs(theo_weights_full).sort(['weights'],ascending=False)
number_series = 5
series_red = sorted_weights.index[0:number_series]

#Training fit - reduced weights
max_sr_train, weights_train = CAPM(rets_train[series_red])
#Do via optim
from scipy.optimize import minimize

def sr_prt(weight,rets):
    '''
    Calc simple -sr given weights and rets vec
    used for optim minisation so return -sr
    '''
    port_rets = rets.dot(weight)
    return -sharpe(port_rets)

x0 = np.linspace(1,1,len(theo_weights_full))/len(theo_weights_full)
res =minimize(fun=sr_prt, x0= x0, args = (rets_train,))

#This should be very close to max_sr_train_full above
sr_calc = -res.fun*np.sqrt(252)

#Work out Min

#Do min variance and efficient frontier

#Compare to Sharpe of DJIA/SP500
rets_bm = Benchmarks.pct_change()
mn_bm = 252*rets_bm.mean()
sd_bm = np.sqrt(252)*rets_bm.std()
sr_bm = np.sqrt(252)*sharpe(rets_bm)
print mn_bm, sd_bm, sr_bm
#calc beta's alpha's 


#do forecast of returns, correlation. Use to Weight
rets.iloc[:,0:10].plot()
###DETOUR TO VAR FORECASTING

from statsmodels.tsa.vector_ar.var_model import VAR, VARResults, VARProcess
import statsmodels
statsmodels.version.version

#Check for NA's in data - have to reduce number of series used as full 30
#gave singular matrix
v1 = VAR(rets_train[series_red], freq='D')
v1.select_order(maxlags=30)
results = v1.fit(5) #From fitted
# results.summary()
results.plot()
# results.plot_acorr()
# plt.show()

#Make forecast for 3months
test_index = rets_test.index
fc_range = pd.date_range(start=test_index[0], periods=2, freq='3M')
fc_periods = len(rets_test[fc_range[0]:fc_range[1]])
lag_order = results.k_ar
fc = results.forecast(rets_train[series_red].values,fc_periods)
fc.shape
fc[:,-1]
df_fc = pd.DataFrame(fc,index=rets.index[0:fc_periods],columns=rets_train[series_red])
df_fc.plot()
plt.show()
df_fc.tail()
cum_rets = ((1+df_fc).cumprod(axis=1)-1)*12/252 #3 month period
cum_rets.head()
cum_rets.plot()
plt.show()

#3m annualise fc of means
mn_fc = cum_rets.iloc[-1]
mn_fc = df_fc.mean()
#forecat vol using ema
vec = rets_train[series_red]
# 
# ewmcov_mat = [ pd.ewmcorr(vec.loc[:,s1],vec.loc[:,s2],span=10)[-1] for s1 in vec.columns for s2 in vec.columns]
# ewmcov_mat = np.array(ewmcov_mat)
# ewmcov_mat.shape
# ewmcov_mat = ewmcov_mat.reshape(10,10)
# ewmcov_mat = pd.DataFrame(ewmcov_mat,
#                           index =vec.columns, 
#                           columns = vec.columns)

def ewmcov( vec, span, cor = True ):
    ''' Generates an exponentially weighted cov/corr matrix from a vector '''
    dim = vec.shape[1]
    if cor:
        ewmcov_mat = [ pd.ewmcorr(vec.loc[:,s1],vec.loc[:,s2],span=span)[-1] for s1 in vec.columns for s2 in vec.columns]
    else:
        ewmcov_mat = [ pd.ewmcov(vec.loc[:,s1],vec.loc[:,s2],span=span)[-1] for s1 in vec.columns for s2 in vec.columns]
    ewmcov_mat = np.array(ewmcov_mat)
    ewmcov_mat = ewmcov_mat.reshape(dim,dim)
    ewmcov_mat = pd.DataFrame(ewmcov_mat,
                              index =vec.columns, 
                              columns = vec.columns)
    return ewmcov_mat

V_train = ewmcov(vec, span=15, cor=False)
R_train = ewmcov(vec, span=15, cor=True)

V_train_std = vec.cov()
R_train_std = vec.corr()

invV = inv(V_train_std)
invR = inv(R_train_std)

weights_fit = invV.dot(mn_fc)
weights_fit /= weights_fit.sum() 
weights_fit

rets_test_fc = rets_test[fc_range[0]:fc_range[1]]
#Check port_fit sr
max_sr_test, weights_test = CAPM(rets_test_fc[series_red])
weights_test.values.tolist()
max_sr_test
sr_fitted = -np.sqrt(252)*sr_prt(weight=weights_fit, rets=rets_test_fc[series_red])
# sr_test = -np.sqrt(252)*sr_prt(weight=weights_test, rets=rets_test_fc[series_red])
sr_fitted
max_sr_test
# weights_test.to_clipboard
weights_test.shape
rets_test_fc[series_red].shape
port_ret = rets_test_fc[series_red].dot(weights_fit)
cumret = (port_ret+1).cumprod()
cumret.plot()
plt.show()
sharpe(port_ret)

rets_test_fc[series_red].mean()*252



#These are very different
sr_fitted
max_sr_test
mn_fc

mn_test = rets_test[series_red].mean()
mn_test*252
((rets_test[series_red]+1).cumprod()).plot()
plt.show()
(mn_fc - mn_test)*252

#Returns forecast
#V, R forecast, can make weights and test



# fci = results.forecast_interval(rets_train[series_red].values[-lag_order:],5)
# results.forecast_interval
x = range(0,fci[1].shape[1])
results.plot_forecast(5)
plt.show()
irf = results.irf(10)
irf.plot(orth=False)
plt.show()
irf.plot(impulse ='DJIA')
plt.show()
irf.plot_cum_effects(orth=False)
plt.show()
fevd = results.fevd(1)
fevd.summary()
fevd.plot()
plt.show()
results.test_causality('DJIA', ['SP500'],kind='f')
results.test_causality('SP500', ['DJIA'],kind='f')
results.test_normality(signif=0.05,verbose=False)

resids = results.resid.sum(axis=1)
resids.plot()
plt.show()
from statsmodels.graphics.gofplots import qqplot, qqline

qqplot(data=resids,line='s')#, dist, distargs, a, loc, scale, fit, line, ax)
plt.show()
from statsmodels.sandbox.tsa.garch import Garch
rets1 = rets_bm.iloc[:,1]
Garch(rets1)

#Getting GARCH working -using RPY2
import rpy2
rpy2.__version__

t =datetime(2013,1,5)
from pandas.tseries.offsets import MonthEnd, BusinessMonthBegin

(t + 2*BusinessMonthBegin())> datetime(2013,1,1)

