import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.core.numeric import dtype
import time
import pickle
import datetime
import logging
import os
import sys
sys.path.append('/home/phcostello/Documents/workspace/FinancePython')
sys.path.append('/home/phcostello/Documents/workspace/FinancePython/DataHandler')
os.chdir('/home/phcostello/Documents/workspace/FinancePython')

from DataHandler.DBReader import DBReader
dbpath = "/home/phcostello/Documents/Data/FinanceData.sqlite"