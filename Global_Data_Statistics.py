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
    tlen=dataset.size
    p=0
    n=0 #due to first
    for i in range(1,dataset.size):
        if dataset[i]>=dataset[i-1]:
            outputfeature.append(1)
            p+=1
        else:
            outputfeature.append(0)
            n+=1
    print("Output Matrix: ")
    print(np.array(outputfeature).size)
    print("\n")
    ##print(outputfeature)
    #print("\n")
    
    return outputfeature,tlen,p,n




def compute_mom_ado_fun(datafile):
    #print("\n\n"+datafile+"\n")
    dataset= pd.read_csv(datafile)
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
    put,tl,p,n=getupdown(open_price_out)
    computed_featureset.append(put)
    
    return computed_featureset,tl,p,n





def compute_stockstat(csvfile):
    dataset=pd.read_csv(csvfile)
    dataset.columns = ['Symbol','Series','Date','Prev Close','	open','high','low','Last Price','close','Average Price','Total Traded Quantity','	Turnover','No. of Trades','Deliverable Qty','% Dly Qt to Traded Qty']
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

#csvfiles=['RELIANCEEQN','INFYEQN','SBINEQN','SUNPHARMAEQN','HDFCEQN','DRREDDYEQN']
csvfiles=['RELIANCEEQN','INFYEQN','HDFCEQN','DRREDDYEQN']
#csvfiles=['RELIANCEEQN']
putyear=2008
putSplitMode=2  # 1 for 20%, 2 for 50%

#------------------------------------
#------------------------------------
#------------------------------------









if(putSplitMode==1):
    putper=20
else:
    putper=50



for onecsvfile in csvfiles:
    years=[]
    peryearmax=[]
    peryearmin=[]
    y=[]
    ttl=[]
    tp=[]
    tn=[]
    pper=[]
    nper=[]
    f=[]
    
    tr=[]
    te=[]
    trp=[]
    trn=[]
    tep=[]
    ten=[]
    
    ff=[]
    mm=[]
    mi=[]
    initial=putyear
    for i in range(10):
        putstr='01-01-'+str(initial)+'-TO-31-12-'+str(initial)+onecsvfile
        #print(putstr)
        y.append(initial)
        years.append(putstr)
        initial=initial+1
    
    for year in years:
        putasmax,putasmin,df,stock=compute_stockstat('Dataset/'+year+'.csv')
        df=df.reset_index()
        mm.append(putasmax)
        mi.append(putasmin)
        print(stock.shape)
        X,tl,n,p=compute_mom_ado_fun('Dataset/'+year+'.csv')    
        ttl.append(tl)
        tp.append(p)
        tn.append(n)
        pper.append((p/tl)*100)
        nper.append((n/tl)*100)
    		
        X=np.array(X).T
        
        df_mad= pd.DataFrame(X,columns=['Momentum','ADOsc', 'OpenPrice', 'UpDown'])
        
        #frames = [df, df_mom_ado]
        #final_result = pd.concat(frames)
        #final=pd.merge(df, df_mom_ado, right_index=True, left_index=True)
        
        #bigdata = df.append(df_mad)
        df_result = pd.concat([df, df_mad], axis=1)
        
        df_result=df_result.iloc[10:,:]
        #df_result.to_csv('.//Dataset//computed_feature_'+year+'.csv', sep=',',index=False) 

        
        if(putSplitMode==1):
            forwhatsplit=10  # --- > to get 10 % training and 10% testing dataset
        else:
            forwhatsplit=2   # --- > to get 50 % training and 50% testing dataset
            
        if (tl/forwhatsplit).is_integer():
            tr.append(tl/forwhatsplit)
            te.append(tl/forwhatsplit)
        else:
            if(putSplitMode==1): # --- > to get 10 % training and 10% testing dataset
                tl=int(tl/5)
            intr=int(tl/2)+1
            inte=tl-intr
            tr.append(intr)
            te.append(inte)
        
        if (p/forwhatsplit).is_integer():
            trp.append(p/forwhatsplit)
            tep.append(p/forwhatsplit)
        else:
            if(putSplitMode==1): # --- > to get 10 % training and 10% testing dataset
                p=int(p/5)
            intr=int(p/2)+1
            inte=p-intr
            trp.append(intr)
            tep.append(inte)
        
        if (n/forwhatsplit).is_integer():
            trn.append(n/forwhatsplit)
            ten.append(n/forwhatsplit)
        else:
            if(putSplitMode==1): # --- > to get 10 % training and 10% testing dataset
                n=int(n/5)
            intr=int(n/2)+1
            inte=n-intr
            trn.append(intr)
            ten.append(inte)
        
        
    
    y.append('Total')
    f.append(y)
    
    numtp=np.array(tp)
    total_tp=sum(numtp)
    tp.append(total_tp)
    f.append(tp)
    
    numpper=np.array(pper)
    total_pper=sum(numpper)/10
    pper.append(total_pper)
    f.append(pper)
    
    numtn=np.array(tn)
    total_tn=sum(numtn)
    tn.append(total_tn)
    f.append(tn)
    
    numnper=np.array(nper)
    total_nper=sum(numnper)/10
    nper.append(total_nper)
    f.append(nper)
    
    numttl=np.array(ttl)
    total_ttl=sum(numttl)
    ttl.append(total_ttl)
    f.append(ttl)
    
    
    
    
    print(['Total',total_tp,total_pper,total_tn,total_nper,total_ttl])
    
    f=np.array(f).T
    
    df_f= pd.DataFrame(f,columns=['Year', 'Increase', '%Increase','Decrease','%Decrease','Total'])
    df_f.to_csv('.//Dataset_Statistics//'+onecsvfile+'_'+str(putyear)+'_OverallYear_Increse_decrese_in_'+str(putper)+'_percent_Training_and_Holdout.csv', float_format='%.2f',sep=',',index=False) 
    
    
    ff=f
    
    f=[]
    f.append(y)
    
    
    numtrp=np.array(trp)
    total_trp=sum(numtrp)
    trp.append(total_trp)
    f.append(trp)
    
    numtrn=np.array(trn)
    total_trn=sum(numtrn)
    trn.append(total_trn)
    f.append(trn)
    
    numtr=np.array(tr)
    total_tr=sum(numtr)
    tr.append(total_tr)
    f.append(tr)
    
    numtep=np.array(tep)
    total_tep=sum(numtep)
    tep.append(total_tep)
    f.append(tep)
    
    numten=np.array(ten)
    total_ten=sum(numten)
    ten.append(total_ten)
    f.append(ten)
    
    numte=np.array(te)
    total_te=sum(numte)
    te.append(total_te)
    f.append(te)
    
    f=np.array(f).T
    df_f= pd.DataFrame(f,columns=['Year', 'Increase', 'Decrease','Total', 'Increase', 'Decrease','Total'])
    df_f.to_csv('.//Dataset_Statistics//'+onecsvfile+'_'+str(putyear)+'_Split_in_'+str(putper)+'_percent_Training_and_Holdout.csv', float_format='%.01f',sep=',',index=False) 
    
    
    print("Done-----------Done-------------Done")
    print("--------------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------------")