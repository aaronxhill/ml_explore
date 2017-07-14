import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')


from numpy import genfromtxt

test_np = genfromtxt('C:/Users/andy/Documents/GitHub/python-module-summarize/sample.csv', delimiter=',')


test_pd = pd.read_csv('C:/Users/andy/Documents/GitHub/python-module-summarize/sample.csv')

nba_pd = pd.read_csv('/Users/Andy/Documents/Python_Project/python-module-summarize/nba_stats.csv')

nba_np = np.genfromtxt('/Users/Andy/Documents/Python_Project/python-module-summarize/nba_stats.csv', delimiter=',')


type(nba_np)

type(nba_pd)


def check_input_data(d):
	""" check that input data is a pandas dataframe or series or a numpy ndarray """
	""" convert input to pandas dataframe (if not one already) """
	""" return error if input is not any of these data structures """

	structure = type(d)

	structure = str(structure)


	if structure == "<class 'pandas.core.frame.DataFrame'>":
		output = 'Data is a Pandas DataFrame'
	elif structure == "<class 'pandas.core.frame.Series'>":
		output = 'Data is a Pandas Series'
	elif structure == "<class 'numpy.ndarray'>":
		output = 'Data is a NumPy Array'
	elif structure == "<class 'dict'>":
		output = 'Data is a Dictionary'
	else: 
		output = 'Data is a List'
	return output

	if structure == "<class 'numpy.ndarray'>":
		new_dataframe = pd.DataFrame(data=d[1:,1:], index=d[1:,0], columns=d[0,1:])
		print("Converted np.array to pd.dataframe")
	else: d
	return 
		
       




data_structures = {
		'pandas.core.frame.DataFrame':0,
		'pandas.core.frame.Series':1,
		'numpy.ndarry':2,
		'list':3,
		'dictionary':4
		}


t = ['t','y','g']
check_data_type('test')

def check_data_type(): #ANDREW
	""" simple function to check data type """

class explore():
	""" summarize and visualize data """

	def __init__(self, data):
		self.data = data

def summarize(data):
	""" method to summarize and visualize self.data """
	types_of_data = data.dtypes
	summary = data.describe(include = 'all')
	bxPlt = data.plot.box(return_type='axes', figsize=(30,30))
	histogram = data.hist(figsize=(20,20))
	return(types_of_data)
	return summary
	print(histogram)
	print(bxPlt)



