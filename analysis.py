from pandas.io.html import read_html
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from dateutil.parser import parse
from prophet import Prophet
from statistics import mean

#getting data from html
def gettingData():
	user_data = pd.read_html("templates/overview.html") #remove
	user_data2 = pd.read_html("templates/transaction.html") #remove

	#html table to dataframe
	cancel_data = user_data[2] #use pandas sql to dataframe  || sql table: 
	sales_data = user_data2[0] #use pandas sql to dataframe || sql table: users_sales
	#rename dataframe columns to the ones below
	
	#cleaning dataframe
	product_code = sales_data['Product Code']
	sales_y = sales_data['Total'].replace({'Php':'',' ':''}, regex=True).astype('float')
	demand_y = sales_data['Quantity'].astype('int')
	string_date = sales_data['Date']
	sales_date = pd.to_datetime(sales_data["Date"])

	parsedDate = [] # quantifiable values of x for data processing
	for date in range(len(string_date)):
		parsed = parse(string_date[date])
		parsedDate.append(int(parsed.strftime('%d')))

	sales_df = pd.DataFrame({'Date':sales_date, 'PDate':parsedDate, 'TotalSales':sales_y, 'ProductCode': product_code})
	demand_df = pd.DataFrame({'Date':sales_date, 'PDate':parsedDate, 'Quantity':demand_y, 'ProductCode': product_code}) 
	cluster_df = pd.DataFrame({'Date':sales_date, 'PDate':parsedDate, 'Quantity':demand_y, 'ProductCode': product_code})

	return sales_df, demand_df, cluster_df, sales_data

# SALES AND DEMAND FORECAST - FSL REGRESSION
def predictData(df):
	m = Prophet()
	m.fit(df)
	future = m.make_future_dataframe(periods=31)
	forecast = m.predict(future)
	pred = forecast['trend']
	trend = forecast['trend'].map(str).values.tolist()
	upper = forecast['yhat_upper'].map(str).values.tolist()
	lower = forecast['yhat_lower'].map(str).values.tolist()

	return trend, lower, upper, pred

# CONFIDENCE LEVEL
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

def getSlope(x,y):
	slope = np.polyfit(x,y,1)[0]
	return slope

# PRODUCT ORDER - CLUSTERING ANALYSIS 
def process_ProductOrder():
	List_df = gettingData()
	cluster_df = List_df[2]
	#list to dataframe 
	agg_functions = {'ProductCode': 'first', 'Quantity': 'mean'}
	clusterdf_new = cluster_df.groupby(cluster_df['ProductCode']).aggregate(agg_functions)
	
	clusterdf_new['MeanDiff'] = clusterdf_new['Quantity'] - clusterdf_new['Quantity'].mean(axis=0)
	n = clusterdf_new['Quantity'].nunique()

	if (n == 1):
		km = KMeans(n_clusters=1,  random_state=0)
	if (n == 2):
		km = KMeans(n_clusters=2,  random_state=0)
	if (n >= 3):
		km = KMeans(n_clusters=3,  random_state=0)

	y_predicted = km.fit_predict(clusterdf_new[['Quantity', 'MeanDiff']])
	swtch = 0
	n = len(set(y_predicted))

	clusterdf_new['Cluster'] = y_predicted
	clustered = clusterdf_new.groupby('Cluster')
	
	if (n == 1):
		fast = clustered.get_group(0)
		fproducts = fast['ProductCode'].map(str).values.tolist()
		sproducts = 0
		nproducts = 0

	if (n == 2):
		swtch = 0
		if (max(clustered.get_group(1)['Quantity']) < max(clustered.get_group(0)['Quantity']) ):
			slow = clustered.get_group(1)
			fast = clustered.get_group(0)
			swtch = 1
		else:
			slow = clustered.get_group(0)
			fast = clustered.get_group(1)
		nproducts = 0
		sproducts = slow['ProductCode'].map(str).values.tolist()
		fproducts = fast['ProductCode'].map(str).values.tolist()

	if(n >= 3):
		swtch = 0
		if (min(clustered.get_group(2)['Quantity']) < min(clustered.get_group(0)['Quantity']) < min(clustered.get_group(1)['Quantity']) ):
			fast = clustered.get_group(1)
			slow = clustered.get_group(0)
			non = clustered.get_group(2)
			swtch = 1
		if(min(clustered.get_group(2)['Quantity']) < min(clustered.get_group(1)['Quantity']) < min(clustered.get_group(0)['Quantity']) ):
			fast = clustered.get_group(0)
			slow = clustered.get_group(1)
			non = clustered.get_group(2)
			swtch = 2
		if(min(clustered.get_group(1)['Quantity']) < min(clustered.get_group(2)['Quantity']) < min(clustered.get_group(0)['Quantity']) ):
			fast = clustered.get_group(0)
			slow = clustered.get_group(2)
			non = clustered.get_group(1)
			swtch = 3
		if(min(clustered.get_group(1)['Quantity']) < min(clustered.get_group(0)['Quantity']) < min(clustered.get_group(2)['Quantity']) ):
			fast = clustered.get_group(2)
			slow = clustered.get_group(0)
			non = clustered.get_group(1)
			swtch = 4
		if(min(clustered.get_group(0)['Quantity']) < min(clustered.get_group(1)['Quantity']) < min(clustered.get_group(2)['Quantity']) ):
			fast = clustered.get_group(2)
			slow = clustered.get_group(1)
			non = clustered.get_group(0)
			swtch = 5
		if(min(clustered.get_group(0)['Quantity']) < min(clustered.get_group(2)['Quantity']) < min(clustered.get_group(1)['Quantity']) ):
			fast = clustered.get_group(1)
			slow = clustered.get_group(2)
			non = clustered.get_group(0)
			swtch = 6

		fproducts = fast['ProductCode'].map(str).values.tolist()
		sproducts = slow['ProductCode'].map(str).values.tolist()
		nproducts = non['ProductCode'].map(str).values.tolist()

	cluster_y = clusterdf_new['Quantity'].map(float).values.tolist()
	cluster_y = [round(num,2) for num in cluster_y]
	
	cluster_x = clusterdf_new['ProductCode'].map(str).values.tolist()
	clusters = clusterdf_new['Cluster'].map(str).values.tolist()

	return fproducts, sproducts, nproducts, cluster_y, cluster_x, clusters, swtch, n, y_predicted

#print(process_ProductOrder()[1])
#print(process_ProductOrder()[2])
#print(process_ProductOrder()[3])
#print(process_ProductOrder()[4])
#print(process_ProductOrder()[5])
#print(process_ProductOrder()[6])
#print(gettingData()[3])