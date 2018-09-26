# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 13:42:17 2018

@author: Dhaval
"""

import pandas as pd
from stockstats import StockDataFrame
import numpy as np

def momentum(dataset):
    momentumfeature = []
    size = dataset.size-9
    
    for i in range(9):
        momentumfeature.append(-1)
    
    for i in range(size):
        pos = i + 9
        momentumfeature.append(dataset[pos] - dataset[i])
    
    print("Momentum Matrix: ")
    print(np.array(momentumfeature).size)
    print("\n")
    ##print(momentumfeature)
          
    return momentumfeature

def A_D_oscillator(close_price, low_price, high_price, volume):
    adoscillatorfeature = []
    size = close_price.size
    
    # previous ADL is taken as 0..
    adl = []
    
    previous_adl = 0
    
    for i in range(size):
        val1 = ((close_price[i]-low_price[i]) - (high_price[i]-close_price[i])) / (high_price[i]-low_price[i])
        #print(type(volume[i]))
        
        vv = volume[i]
        
        val2 = val1 * vv
        adl_ = previous_adl + val2
        adl.append(adl_)
        previous_adl = adl_


    d = {'close':close_price, 'low':low_price, 'high':high_price, 'adl':adl}
    df = pd.DataFrame(d)
    
    stock = StockDataFrame.retype(df)    
    
    res1 = stock['adl_3_ema']
    res2 = stock['adl_10_ema']
    
    for i in range(9):
        adoscillatorfeature.append(-1)
        
    for i in range(size-9):
        adoscillatorfeature.append(res1[i] - res2[i])
    
    print("A/D Oscillator Matrix: ")
    print(np.array(adoscillatorfeature).size)
    print("\n")
    
    
    """
    ema_3 = []
    ema_10 = []
    
    # for 3..
    mul = 2 / (3 + 1)
    previous_ema = 0
    
    for i in range(3):
        previous_ema += adl[i + 7]
    previous_ema /= 3
    
    ema_3.append(previous_ema)
    
    for i in range(10,size):
        val = (adl[i] - previous_ema) * mul + previous_ema
        ema_3.append(val)
        previous_ema = val
    
    
    # for 10..
    mul = 2 / (10 + 1)
    previous_ema = 0
    
    for i in range(10):
        previous_ema += adl[i]
    previous_ema /= 10
    
    ema_10.append(previous_ema)
    
    for i in range(10,size):
        val = (adl[i] - previous_ema) * mul + previous_ema
        ema_10.append(val)
        previous_ema = val
    
    
    # calculate..
    for i in range(size-9):
        val = ema_3[i] - ema_10[i]
        adoscillatorfeature.append(val)
    
    print("A/D Oscillator Matrix: ")
    print(np.array(adoscillatorfeature).size)
    print("\n")
    ##print(adoscillatorfeature)
    #print("\n")
    """
    return adoscillatorfeature

def getupdown(dataset):
    ##print(dataset)
    outputfeature=[]
    outputfeature.append(0)
    
    for i in range(1,dataset.size):
        if dataset[i]>=dataset[i-1]:
            outputfeature.append(1)
        else:
            outputfeature.append(0)
    print("Output Matrix: ")
    print(np.array(outputfeature).size)
    print("\n")
    ##print(outputfeature)
    #print("\n")
    
    return outputfeature




def compute_mom_ado_fun(datafile):
    #print("\n\n"+datafile+"\n")
    dataset= pd.read_csv('.//Dataset//'+datafile)
    filtered_dataset=dataset.iloc[:,3:]
    output_open_price=dataset.iloc[:,4:5]
    output_open_price_dataset=dataset.iloc[:,4:5]
    computed_featureset=[]
    
    open_price=np.array(filtered_dataset['Open Price'])
    low_price=np.array(filtered_dataset['Low Price'])
    high_price=np.array(filtered_dataset['High Price'])
    close_price = np.array(filtered_dataset['Close Price'])
    

    volume = np.array(filtered_dataset['Total Traded Quantity'])
    
    """
    total_traded = np.array(filtered_dataset['Total Traded Quantity'])
    deliver = np.array(filtered_dataset['Deliverable Qty'])
    
    for i in range(deliver.shape[0]):
        deliver[i] = float(deliver[i])
    
    volume = deliver / total_traded
    """
    
    
    computed_featureset.append(momentum(close_price))
    computed_featureset.append(A_D_oscillator(close_price, low_price, high_price, volume))
    
    
    
    ##print(output_open_price_dataset)
    open_price=np.array(output_open_price).T
    ##print(open_price.shape)
    open_price=open_price.tolist()
    ##print(open_price[0])    
    computed_featureset.append(open_price[0])
    
    
    ##print(output_open_price_dataset)
    open_price_out=np.array(output_open_price_dataset)
    #getupdown(open_price_out)
    computed_featureset.append(getupdown(open_price_out))
    
    return computed_featureset





def compute_stockstat(csvfile):
    dataset=pd.read_csv('.//Dataset//'+csvfile)
    dataset.columns = ['Symbol','Series','Date','Prev Close','	open','high','low','Last Price','close','Average Price','Total Traded Quantity','Turnover','No. of Trades','Deliverable Qty','% Dly Qt to Traded Qty']
    stock = StockDataFrame.retype(dataset)    
    
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    
    result=stock['close_10_sma']
    
    result=stock['close_10_ema']
    #momentum
    result=stock['kdjk_10']
    
    result=stock['kdjd_10']
    
    result=stock['macd']
    
    print(result.size)
    
    result=stock['rsi_10']
    
    result=stock['wr_10']
    
    #a/d oc
    result=stock['cci_10']
    
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    result=np.array(result)
    rel_max=np.nanmax(result)
    rel_min=np.nanmin(result)
    
    
    df = stock[['close_10_sma', 'close_10_ema','kdjk_10','kdjd_10','macd','rsi_10','wr_10','cci_10']]
    
    return rel_max,rel_min,df,stock

#------------------------------------
#------------------------------------
#------------------------------------
yearToStart=2008
#csvfiles=['RELIANCEEQN','INFYEQN','SBINEQN','SUNPHARMAEQN','HDFCEQN','DRREDDYEQN']
csvfiles=['RELIANCEEQN','INFYEQN','HDFCEQN','DRREDDYEQN']

#------------------------------------
#------------------------------------
#------------------------------------

for onecsvfile in csvfiles:
    years=[]
    peryearmax=[]
    peryearmin=[]
    
    initial=yearToStart
    for i in range(10):
        putstr='01-01-'+str(initial)+'-TO-31-12-'+str(initial)+onecsvfile
        #print(putstr)
        years.append(putstr)
        initial=initial+1
    
    for year in years:
        putasmax,putasmin,df,stock=compute_stockstat(year+'.csv')
        df=df.reset_index()
        
        print(stock.shape)
        X=compute_mom_ado_fun(year+'.csv')    
        X=np.array(X).T
        
        df_mad= pd.DataFrame(X,columns=['Momentum','ADOsc', 'OpenPrice', 'UpDown'])
        
        #frames = [df, df_mom_ado]
        #final_result = pd.concat(frames)
        #final=pd.merge(df, df_mom_ado, right_index=True, left_index=True)
        
        #bigdata = df.append(df_mad)
        df_result = pd.concat([df, df_mad], axis=1)
        
        df_result=df_result.iloc[10:,:]
        df_result.to_csv('.//Dataset//'+'computed_feature_'+year+'.csv', sep=',',index=False) 
print("Done-----------Done-------------Done")
print("--------------------------------------------------------------------------------------")
print("--------------------------------------------------------------------------------------")

