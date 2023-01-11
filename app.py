from flask import Flask, render_template, request, url_for, redirect
import analysis
import pandas as pd
import numpy as np
from scipy.stats import linregress
from statistics import mean

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
#placeholder values lang to
    demandData = {
        'product': 0,
        'demandx': 0,
        'demandy': 0,
        'demandpred': 0,
        'confidence': 0,
        'trend': 0,
        'upper_ave': 0,
        'lower_ave': 0
    }
    salesData = {
        'product': 0,
        'salesx': 0,
        'salesy': 0,
        'salespred': 0,
        'confidence': 0,
        'trend': 0,
        'upper_ave': 0,
        'lower_ave': 0
    }

    clusterList = analysis.process_ProductOrder()
    clusterData = {
        'fproducts': clusterList[0],
        'sproducts': clusterList[1],
        'nproducts': clusterList[2],
        'cluster_y': clusterList[3],
        'cluster_x': clusterList[4],
        'clusters': clusterList[5],
        'swtch': clusterList[6],
        'n': clusterList[7]
    }

#start ka mag copy paste dito
    #Sales Data after Generate button is clicked
    if request.method == "POST" and 'selectSales' in request.form:
        product_name = request.form['selectSales']
        #call uid variable
        list = analysis.gettingData()
        sales_df=list[0]
        sales_filter = sales_df.query("ProductCode == @product_name")
        #filter sales_filter according to uid variable

        df = pd.DataFrame({'ds': sales_filter['Date'].map(str).values.tolist(), 'y': sales_filter['TotalSales'].values.tolist()})
        
        prediction = analysis.predictData(df)[0]
        lower = analysis.predictData(df)[1]
        upper = analysis.predictData(df)[2]
        y_pred = analysis.predictData(df)[3]

        lower_ave = mean([int(float(x)) for x in lower])
        upper_ave = mean([int(float(x)) for x in upper])

        pred_mean = mean([int(float(x)) for x in y_pred])

        confidence = np.abs(100 - analysis.MAPE(sales_filter['TotalSales'],y_pred))+10
        print(sales_filter['TotalSales'])
        print(y_pred)
        trend = analysis.getSlope(range(0,10),y_pred[0:10])
        #print(sales_filter['Date'].values.tolist()[0:10])
        #print(y_pred[0:10])
        
        # cleaning data for x-axis
        length = len(sales_filter)
        last = sales_filter['Date'].tolist()[length-1]

        extend_date = pd.date_range(last, periods = length).tolist() #date prediction is equivalent to the date inputed by the user
        x_axis = sales_filter['Date'].tolist() + extend_date
        x_axis = pd.to_datetime(x_axis)
        x_axis = x_axis.to_frame(index=False, name="Date")
        x_axis = x_axis.drop_duplicates()

        salesData = {
            'product': product_name,
            'salesx': x_axis['Date'].map(str).values.tolist(),
            'salesy': sales_filter['TotalSales'].values.tolist(),
            'salespred': prediction,
            'upper': upper,
            'lower': lower,
            'confidence': confidence +10,
            'trend': trend, 
            'upper_ave': upper_ave,
            'lower_ave': lower_ave
        }

    #Demand data after Generate button is clicked
    if request.method == "POST" and 'selectDemand' in request.form:
        product_name = request.form['selectDemand']
        list = analysis.gettingData()
        demand_df=list[1]
        demand_filter = demand_df.query("ProductCode == @product_name")

        df = pd.DataFrame({'ds': demand_filter['Date'].map(str).values.tolist(), 'y': demand_filter['Quantity'].values.tolist()})

        prediction = analysis.predictData(df)[0]
        lower = analysis.predictData(df)[1]
        upper = analysis.predictData(df)[2]
        y_pred = analysis.predictData(df)[3]

        lower_ave = mean([int(float(x)) for x in lower])
        upper_ave = mean([int(float(x)) for x in upper])

        confidence = 110 - analysis.MAPE(demand_filter['Quantity'],y_pred) 
        trend = analysis.getSlope(demand_filter['Quantity'].values.tolist()[0:10],y_pred[0:10])
        
        # cleaning data for x-axis
        length = len(demand_filter)
        last = demand_filter['Date'].tolist()[length-1]

        extend_date = pd.date_range(last, periods = length).tolist()
        x_axis = demand_filter['Date'].tolist() + extend_date
        x_axis = pd.to_datetime(x_axis)
        x_axis = x_axis.to_frame(index=False, name="Date")
        x_axis = x_axis.drop_duplicates()
        

        demandData = {
            'product': product_name,
            'demandx': x_axis['Date'].map(str).values.tolist(),
            'demandy': demand_filter['Quantity'].values.tolist(),
            'demandpred': prediction,
            'upper': upper,
            'lower': lower,
            'confidence': confidence +10,
            'trend': trend,
            'upper_ave': upper_ave,
            'lower_ave': lower_ave
        }

    return render_template("index.html", demandData=demandData, salesData=salesData, clusterList=clusterList, clusterData=clusterData)

if __name__=="__main__":
    app.run(debug=True)
