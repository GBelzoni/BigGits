# Put any initialization logic here.  The context object will be passed to
# the other methods in your algorithm.
import pandas as pd
from datetime import datetime

def initialize(context):
    
    #Get data
    # sid(21839)
    context.tickers = { 'Verizon': sid(21839),
               'HewlettPackard': sid(3735),
               'JohnsonJohnson': sid(4151),
               'Boeing': sid(698),
               'AmericanExpress': sid(679)
               }
    
    #import weights
    weights = [0.1333333333,
                       0.3333333333,
                       0.0666666667,
                       0.2666666667,
                       0.2]
    
    cols = ['AmericanExpress',
            'HewlettPackard',
            'JohnsonJohnson',
            'Boeing',
            'Verizon']

    date = datetime(2013,4,1)
    
    context.weights = pd.DataFrame(weights,columns=[date],index=cols)
    #construct weight mapping
    
    
    

    
 
# Will be called on every trade event for the securities you specify. 
def handle_data(context, data):
    # Implement your algorithm logic here.

    # data[sid(X)] holds the trade event data for that security.
    # data.portfolio holds the current portfolio state.
    
    
    # Place orders with the order(SID, amount) method.

    # TODO: implement your own logic here.
    
    print context.tickers
    
    #For time 0
    
    #get portfolio value
    
    #Work out value for each ticker for portfolio
    
    #Get current value position - for time >0
    
    #Workout how much to change current position by
    
    #place orders
    
    
    
    # order(sid(24), 50)