'''
Created on Aug 4, 2013

@author: phcostello
'''
from BoilerPlate import *

from statsmodels.tsa.kalmanf import KalmanFilter
from statsmodels.tsa.kalmanf.kalmanfilter import kalmanfilter
#from FinancePython.Examples.kalmanfilter import kalmanfilter

if __name__ == '__main__':
   
    def example_from_web():
        # from http://www.ilovefpv.com/modelling-kalman-filter-python-scipy-numpy-libraries/
        
        ################################################################################
        #
        # Filename: kalman_2012xxxx.py
        #
        # Description - implements the Kalman filter example as provided on Wikipedia
        # page: http://en.wikipedia.org/wiki/Kalman_filter
        #
        # Example uses a 400Hz sampling rate with noise applied to both measurement
        # and acceleration.
        #
        ################################################################################
        from pylab import *
        import random
        from numpy import array, dot
        # Sampling rate: 400Hz
        sampling_rate = 1.0/400.0
        
        # Simulation duration (seconds)
        simulation_duration = 120
        number_of_samples = int(simulation_duration / sampling_rate)
        
        # Noise parameters
        # sigma a_k
        # sigma_z
        sigma_a_k = 2.0
        sigma_z = 0.2
        
        # State transition model as applied to x_k
        F = array([[1.0, sampling_rate],
        [0.0, 1.0] ])
        
        G = array([[sampling_rate**2.0/2.0],
        [sampling_rate]])
        
        # Covariance Matrix
        Q_k = dot(G,G.T)*sigma_a_k**2.0
        
        # Measurement to state translation - in this case we directly measure position
        H_k = array([1.0, 0.0])
        
        # Observation noise covariance
        R_k = sigma_z**2
        
        # Initial Conditions
        x_k = array([[0.0],
        [0.0]])
        x_km1 = array([[0.0],
        [0.0]])
        xh_k_km1 = array([[0.0],
        [0.0]])
        xh_km1_km1 = array([[0.0],
        [0.0]])
        P_km1_km1 = array([[0.0, 0.0],
        [0.0, 0.0]])
        
        # Reserve log vectors for plotting later
        x_log = zeros((2,number_of_samples))
        z_log = zeros((2,number_of_samples))
        yh_log = zeros((2,number_of_samples))
        xh_log = zeros((2,number_of_samples))
        k_log = zeros((2,number_of_samples))
        P_log = zeros((2,number_of_samples))
        
        # Sampling Loop
        for t in range(1,number_of_samples):
            if t%1000 == 0:
                print t
        
            a_k = random.normalvariate(0.1*sin(2*pi*t*sampling_rate), sigma_a_k) # Random acceleration
            x_k = dot(F,x_km1) + dot(G,a_k) # The real state
            
            # Predict
            xh_k_km1 = dot(F,xh_km1_km1) # Predicted state estimate
            P_k_km1 = dot(dot(F,P_km1_km1), F.T) + Q_k # Predicted covariance
            
            # Measurement
            z_k = dot(H_k, x_k) + random.normalvariate(0.0, sigma_z+t/(10000.0))
            
            # Update
            yh_k = z_k - dot(H_k, xh_k_km1) # Difference between measurement and prediction
            
            S_k = dot(dot(H_k,P_k_km1),H_k.reshape(2,1)) + R_k # Residual covariance
            K_k = dot(P_k_km1, H_k.reshape(2,1))*S_k # Kalman Gain
            
            xh_k_k = xh_k_km1 + K_k*yh_k # Updated state estimate
            
            P_k_k = (1 - dot(H_k, K_k))*P_k_km1 # Updated covariance estimate
            # ~~~~~~~~~~~~~ This is a reversal of the published equation...
            
            # Save for next cycle
            x_km1 = x_k
            xh_km1_km1 = xh_k_k
            P_km1_km1 = P_k_k
            
            # Data logging
            x_log[0,t] = x_k[0,0]
            x_log[1,t] = x_k[1,0]
            z_log[0,t] = z_k
            xh_log[0,t] = xh_k_k[0,0]
            k_log[0,t] = K_k[1]
            yh_log[0,t] = yh_k
            P_log[0,t] = P_k_k[0,0]
            P_log[1,t] = P_k_k[1,1]
        
        # Figure 1
        plot(x_log[0,:])
        #plot(z_log[0,:])
        plot(xh_log[0,:])
        xlabel('Sample')
        ylabel('Position (meters)')
        title('Position : Actual State')
        grid(True)
        
        # Figure 2
        figure()
        xlabel('Sample')
        ylabel('Velocity (meters/second)')
        title('Velocity : Actual State')
        grid(True)
        plot(x_log[1,:])
        
        # Figure 2
        figure()
        xlabel('Sample')
        title('Kalman Position Gain over Time')
        grid(True)
        plot(k_log[0,:])
        
        # Figure 3
        figure()
        xlabel('Sample')
        title('Residual over Time')
        grid(True)
        plot(yh_log[0,:])
        
        # Figure 4
        figure()
        xlabel('Sample')
        title('Covariances')
        grid(True)
        plot(P_log[0,:])
        plot(P_log[1,:])
        
        show()
    
    def example_SP500():
        
        from DataHandler.DBReader import DBReader
        dbr = DBReader()
        data = dbr.readSeries('SP500')
        
        data['Adj_Close'].plot()
        data.info()
#         plt.show()
        
        #Def F, 
        
    def example_Nile():
        
        #This example shows that Kalman filter and ewma are very similar in simple case.
        #Note that Kalman can give confidence interval estimates, might be good for calibrating
        #Entry exit signals.
        
        #Need to investigate how to dynamically estimate parameters - see DLM book, seems to need MCMC to approximate
        #distributions of parameters
        
        #Another idea might be to use GARCH type assumption on volatility somehow.
        
#        from BoilerPlate import *        
        NileData = pd.read_csv('Nile.csv')
        NileLevel = NileData['Level'].values        
        NileData = NileData.set_index('Date')
        
        
        #This example uses the statsmodel Kalman filter
        """
        Returns the negative log-likelihood of y conditional on the information set
        
        Assumes that the initial state and all innovations are multivariate
        Gaussian.
        
        Parameters
        -----------
        F : array-like
        The (r x r) array holding the transition matrix for the hidden state.
        A : array-like
        The (nobs x k) array relating the predetermined variables to the
        observed data.
        H : array-like
        The (nobs x r) array relating the hidden state vector to the
        observed data.
        Q : array-like
        (r x r) variance/covariance matrix on the error term in the hidden
        state transition.
        R : array-like
        (nobs x nobs) variance/covariance of the noise in the observation
        equation.
        y : array-like
        The (nobs x 1) array holding the observed data.
        X : array-like
        The (nobs x k) array holding the predetermined variables data.
        xi10 : array-like
        Is the (r x 1) initial prior on the initial state vector.
        ntrain : int
        The number of training periods for the filter.  This is the number of
        observations that do not affect the likelihood.
        
        
        Returns
        -------
        likelihood
        The negative of the log likelihood
        history or priors, history of posterior
        If history is True.
        
        Notes
        -----
        No input checking is done.
        """
        # uses log of Hamilton 13.4.1
        F = 0.996 #state transition matrix
        A = 0 #control matrix
        H = 1 #state-observed matrix
        Q = 1 #error covariance for state
        R = 10 #error covariance for observation/state
        
        y = NileLevel
        X = 0
        xi10 = y[0]
        ntrain = 0
        history = True
        
        likelyhood, res = kalmanfilter(F, A, H, Q, R, y, X, xi10, ntrain, history )
        
        print likelyhood
        print res
#         print K
        series = res.reshape(100)
        
        
        dfRes = pd.DataFrame(series, index=NileData.index)
        dfRes = dfRes.shift(-1)
        ma = pd.ewma(dfRes,0.1)
        
        dfDatRes = pd.merge(dfRes,ma,how='inner',left_index=True, right_index=True)
        dfDatRes = pd.merge(dfDatRes,NileData,how='inner',left_index=True, right_index=True)
        
        dfDatRes.plot()
        plt.show()
        
    example_Nile()
    #example_SP500()      