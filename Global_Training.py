# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 23:48:59 2018

@author: Dhaval
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 12:39:47 2018

@author: Dhaval
"""
import keras
import os
import scipy
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_mldata
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os
import random
 


def get_run_time(stime,etime):
    millis=etime-stime
    allmilli=millis
    seconds=int(millis/1000)
    minutes=int(seconds/60)
    hours=int(minutes/60)
    puth=hours%24
    putm=minutes%60
    puts=seconds%60
    putmilli=millis-(seconds*1000)
    putstr="Time to train -->   "+str(puth)+" Hours :  "+str(putm)+" Minutes : "+str(puts)+" Seconds : "+str(putmilli)+" Milliseconds"
    return putstr,puth,putm,puts,putmilli,allmilli


def saveAllInCSV(putyear,putcmp,putmodel,node,epoch,companyName,EfficiencyToSave,PrecisionToSave,RecallToSave,FMeasureToSave,HHToSave,MMToSave,SSToSave,MSToSave,AllMSToSave):
    EfficiencyToSave=np.array(EfficiencyToSave)
    PrecisionToSave=np.array(PrecisionToSave)
    RecallToSave=np.array(RecallToSave)
    FMeasureToSave=np.array(FMeasureToSave)
    HHToSave=np.array(HHToSave)
    MMToSave=np.array(MMToSave)
    SSToSave=np.array(SSToSave)
    MSToSave=np.array(MSToSave)
    AllMSToSave=np.array(AllMSToSave)
    FinalSave=[]
    for i in range(9):
        temp=[]
        temp.append(EfficiencyToSave[i])
        temp.append(PrecisionToSave[i])
        temp.append(RecallToSave[i])
        temp.append(FMeasureToSave[i])
        temp.append(HHToSave[i])
        temp.append(MMToSave[i])
        temp.append(SSToSave[i])
        temp.append(MSToSave[i])
        temp.append(AllMSToSave[i])
        FinalSave.append(temp)    
    FinalSave=np.array(FinalSave)
    FinalSave=FinalSave.transpose()
    
    putfold='./'+str(putyear)+'/'+putcmp+'/'+putmodel+'/'
    path = putfold+'Results/Epoch_'+str(epoch)+'/Nodes_'+str(node)+'/'   # if folder doesn't exists then create new folder
    print("Made folder : "+putfold)
    print("Made folder : "+path)
    
    import os
    if not os.path.exists(path):
        os.makedirs(path)  
        
    np.savetxt(path+'Nodes_'+str(node)+'_Epoch_'+str(epoch)+'_All_Efficiency_Of_Testing.csv',FinalSave, fmt='%.10f',delimiter=',', header='Momentum01,02,03,04,05,06,07,08,09')
    

def savemodel(stime,etime,allnu,nu,onecsvfile,onecm,oneeffi,noofnodes,noofepoch,noofbatchsize,noofsplitinratio):    
    saveFile = open(allnu+'_Of_Testing.txt','a')
    saveFile.write("\n"+nu)    
    saveFile.write("\n\n\t"+onecsvfile)
    saveFile.write("\n\t")
    saveFile.write(str(onecm[0])+"\n\t")
    saveFile.write(str(onecm[1]))
    saveFile.write("\n\n")
    saveFile.write("\n\t\tEfficiency : "+str(oneeffi)+"\n")
    saveFile.write("\n\n")
    putstr,puth,putm,puts,putmilli,putallmilli=get_run_time(stime,etime)
    saveFile.write("\n\t\t "+putstr+"\n")
    saveFile.write("\n\n--------------------------------------------------------------------------\n\n")
    saveFile.close()


def GiveFoldersAccordingToCustomChoice(putyear,putcmp,putmodel,putcompany,putcustom,putactivation,putoptimizer,putlr,putmc,noofsplitinratio,noofbatchsize,noofnodes,noofepoch):
    
    # NodesAndEpochs  or   OptimizerMethod   or Company  
    p2='Epoch'+str(noofepoch)+'/Node'+str(noofnodes)+'/MC'+str(putmc*100)  
    #p2='Updated_Node'+str(noofnodes)+'_Epoch'+str(noofepoch)  
    p3='ANN'
    
    if(putcustom):
        p3=p3+'_C'
        if(putoptimizer=='sgd'):
            p3=p3+'_SGD'
            putinnu='_LR'+str(int(putlr*100))+'MC'+str(int(putmc*100))+'SR1to'+str(noofsplitinratio)+'_BSZ'+str(noofbatchsize)
            putfoldername='/'+p3+'/'+p2+'/'
        elif(putoptimizer=='adam'):
            p3=p3+'_Adam'
            putinnu='_LR'+str(int(putlr*100))+'MC_NotDefine_'+'SR1to'+str(noofsplitinratio)+'_BSZ'+str(noofbatchsize)
            putfoldername='/'+p3+'/'+p2+'/' 
    else:
        p3=p3+'_default'
        p3=p3+'_'+putoptimizer
        if(putoptimizer=='adam'):
            putinnu='_LR_0_001'+'MC_NotDefine_'+'SR1to'+str(noofsplitinratio)+'_BSZ'+str(noofbatchsize)
        else:    
            putinnu='_LR_NotDefine'+'MC_NotDefine_'+'SR1to'+str(noofsplitinratio)+'_BSZ'+str(noofbatchsize)
        putfoldername='/'+p3+'/'+p2+'/' 


    putfold='./'+str(putyear)+'/'+putcmp+'/'+putmodel+putfoldername
    print("Made folder : "+putfold)
    putfold_for_csv="./Dataset/"
    
    return putfold_for_csv,putfold,putinnu

def get_one_year_filter_data(year,noofnodes,noofepoch,noofbatchsize,noofsplitinratio):
    #print(year+"\n--------")
    
    #import os
    #os.chdir("Stock-Trend-Prediction-Google-Colab")
    dataset= pd.read_csv(year+'.csv')
    
    
    #print(dataset.shape)
    num_dataset=np.array(dataset)
    bool_positive = (num_dataset[:,12] == 1)
    bool_negative = (num_dataset[:,12] == 0)
    positive_dataset=dataset[bool_positive]
    negative_dataset=dataset[bool_negative]
    #print(positive_dataset.shape)
    #print(negative_dataset.shape)
    positive_dataset_input=positive_dataset.iloc[:,0:12]
    positive_dataset_output=positive_dataset.iloc[:,12:]
    #print(positive_dataset_input.shape)
    #print(positive_dataset_output.shape)
    pos_train_in, pos_test_in, pos_train_out, pos_test_out = train_test_split(positive_dataset_input,positive_dataset_output, test_size=4/5)
    pos_train_in, pos_test_in, pos_train_out, pos_test_out = train_test_split(pos_train_in,pos_train_out, test_size=1/noofsplitinratio)
    #print(year)
    #print(pos_train_in.shape)
    #print(pos_test_in.shape)
    negative_dataset_input=negative_dataset.iloc[:,0:12]
    negative_dataset_output=negative_dataset.iloc[:,12:]
    #print(negative_dataset_input.shape)
    #print(negative_dataset_output.shape)
    neg_train_in, neg_test_in, neg_train_out, neg_test_out = train_test_split(negative_dataset_input,negative_dataset_output, test_size=4/5)
    neg_train_in, neg_test_in, neg_train_out, neg_test_out = train_test_split(neg_train_in,neg_train_out, test_size=1/noofsplitinratio)
    #print(neg_train_in.shape)
    #print(neg_test_in.shape)
       
    one_train_in=np.concatenate([pos_train_in, neg_train_in])
    one_test_in=np.concatenate([pos_test_in, neg_test_in])
    one_train_out=np.concatenate([pos_train_out, neg_train_out])
    one_test_out=np.concatenate([pos_test_out, neg_test_out])
    
    return one_train_in,one_test_in,one_train_out,one_test_out


def get_dataset(putfold_for_csv,stockname,onecsvfile,noofnodes,noofepoch,noofbatchsize,noofsplitinratio,putyear):
    #sharetype='EQN'
    years=[]
    initial=putyear
    for i in range(10):
        putstr=putfold_for_csv+'computed_feature_01-01-'+str(initial)+'-TO-31-12-'+str(initial)+onecsvfile
        
        #print(putstr)
        years.append(putstr)
        initial=initial+1
        
    all_train_in,all_test_in,all_train_out,all_test_out=get_one_year_filter_data(years[0],noofnodes,noofepoch,noofbatchsize,noofsplitinratio)
    #print(all_train_out.shape)
    for year in range(1,len(years)):
        #print(years[year])
        #print(neg_test_out)
        #print(neg_test_in)
        one_train_in,one_test_in,one_train_out,one_test_out=get_one_year_filter_data(years[year],noofnodes,noofepoch,noofbatchsize,noofsplitinratio)
        #print(one_train_out.shape)
        #print(all_train_out.shape)
        
        #np.concatenate([x, y,z])
        all_train_in=np.concatenate([all_train_in, one_train_in])
        all_test_in=np.concatenate([all_test_in, one_test_in])
        all_train_out=np.concatenate([all_train_out, one_train_out])
        all_test_out=np.concatenate([all_test_out, one_test_out])
        
    '''print(all_train_in.shape)
    print(all_test_in.shape)
    print(all_train_out.shape)
    print(all_test_out.shape)
    '''
    
    '''all_train_in=all_train_in.dropna()
    all_test_in=all_test_in.dropna()
    all_train_out=all_train_out.dropna()
    all_test_out=all_test_out.dropna()
    '''
    return all_train_in,all_test_in,all_train_out,all_test_out
    
def save_data_mapping_with_dates(train_in,test_in,train_out,test_out):
    #Completing Save
    #all_train_out=np.concatenate([train_in, train_out])
    tem_train_in=pd.DataFrame(train_in)
    tem_test_in=pd.DataFrame(test_in)
    tem_train_out=pd.DataFrame(train_out)
    tem_test_out=pd.DataFrame(test_out)
    
    
    all_train=pd.concat([tem_train_in, tem_train_out], axis=1)
    all_test=pd.concat([tem_test_in, tem_test_out], axis=1)
    putheaders=['date','close_10_sma','close_10_ema','kdjk_10','kdjd_10','macd','rsi_10','wr_10','cci_10','Momentum','ADOsc','OpenPrice','UpDown']
    all_train.columns =putheaders
    all_test.columns =putheaders
    all_train.to_csv(putfoldername+'Mapped_Train_data_with_dates.csv',index=False)
    all_test.to_csv(putfoldername+'Mapped_Test_data_with_dates.csv',index=False)
    without_date_train_in=train_in[:,1:12]
    without_date_test_in=test_in[:,1:12]
    
    return without_date_train_in,without_date_test_in,train_out,test_out
    
            

def compute_effi(putmodelindex,putfoldername,putcustom,putoptimizer,putactivation,putlr,putmc,csvfile,allnu,nu,train_in,test_in,train_out,test_out,noofnodes,noofepoch,noofbatchsize,noofsplitinratio):
    
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(-1, 1))
    train_in_sliced=scaler.fit_transform(train_in[:,0:10])
    test_in_sliced=scaler.transform(test_in[:,0:10])
    
    final_train_in=[]
    final_test_in=[]
    
    for i in range(train_in_sliced.shape[0]):
        onerow=[]
        for j in range(10):
            onerow.append(train_in_sliced[i][j])
        onerow.append(train_in[i][10])
        final_train_in.append(onerow)
        
    for i in range(test_in_sliced.shape[0]):
        onerow=[]
        for j in range(10):
            onerow.append(test_in_sliced[i][j])
        onerow.append(test_in[i][10])
        final_test_in.append(onerow)
    
    final_train_in=np.array(final_train_in)
    final_test_in=np.array(final_test_in)  
      
    putheaders='close_10_sma,close_10_ema,kdjk_10,kdjd_10,macd,rsi_10,wr_10,cci_10,Momentum,ADOsc,OpenPrice'

    np.savetxt(putfoldername+'Data_TrainingInput.csv',final_train_in, fmt='%.10f',delimiter=',', header=putheaders)
    np.savetxt(putfoldername+'Data_TrainingOutput.csv',train_out, fmt='%.10f',delimiter=',', header='UpDown')
    np.savetxt(putfoldername+'Data_TestingInput.csv',final_test_in, fmt='%.10f',delimiter=',', header=putheaders)
    np.savetxt(putfoldername+'Data_TestingOutput.csv',test_out, fmt='%.10f',delimiter=',', header='UpDown')
    
    train_dataset= pd.read_csv(putfoldername+'Data_TrainingInput.csv')
    corel_full_matrix=train_dataset.corr(method='pearson')
    OpenPriceRelation=corel_full_matrix['OpenPrice']
    OpenPriceRelation=np.array(OpenPriceRelation)
    OpenPriceRelationFilter = (OpenPriceRelation!=1)
    
    
    #print(corel_full_matrix.shape)
    
    train_in=final_train_in[:,OpenPriceRelationFilter]
    test_in=final_test_in[:,OpenPriceRelationFilter]
    
    #print(train_in.shape)
    #print(test_in.shape)
    
    # Initialising the ANN
    classifier = Sequential()    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(activation = putactivation,kernel_initializer = 'uniform',input_dim = 10,units = noofnodes))
    # Adding the output layer
    classifier.add(Dense( kernel_initializer = 'uniform',activation = 'sigmoid',units= 1))
    # Compiling the ANN
    
    #print('--------')
    
    if(putcustom):
        if(putoptimizer=='sgd'):
                sgd = optimizers.SGD(lr=putlr, momentum=putmc)
                classifier.compile(optimizer=sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])   
        elif(putoptimizer=='adam'):
            adm=optimizers.Adam(lr=putlr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            classifier.compile(optimizer=adm, loss = 'binary_crossentropy', metrics = ['accuracy'])
    else:
       classifier.compile(optimizer=putoptimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
      
    ##################################################
    ##################################################
    ##################################################
    
    if(putmodelindex==1):     #Randommodel
        # Do nothing
        # In Random Model, So no need to use set_weights method
        timepass=0   # just for fun  :) 
        timepass=timepass+1 # just for fun  :)
        #print("In Random Model, So no need to use set_weights method "+str(timepass))
    elif(putmodelindex==2):    # Pearson
        final = []  
        inp_hidden = []
        hidden_bias = []
        hidden_op = []
        op_bias = []  
        for i in range(10):   # put pearson coeffi as weights on edges
            temp = []
            for j in range(noofnodes):
                temp.append(OpenPriceRelation[i])
            inp_hidden.append(temp)
        hidden_bias=np.random.uniform(-1,1,size=(noofnodes,))       # random bais from range -1 to 1 
        hidden_op = np.random.uniform(-1, 1, size=(noofnodes,1))   
        op_bias = np.random.uniform(-1, 1, size=(1,))
        final.append(np.array(inp_hidden))
        final.append(np.array(hidden_bias))
        final.append(np.array(hidden_op))
        final.append(np.array(op_bias))
        classifier.set_weights(final)
    elif(putmodelindex==3):    # Pearson ABSOLUTE
        final = []  
        inp_hidden = []
        hidden_bias = []
        hidden_op = []
        op_bias = []
        for i in range(10):   # put pearson coeffi as weights on edges
            temp = []
            for j in range(noofnodes):
                temp.append(abs(OpenPriceRelation[i]))
            inp_hidden.append(temp)
        hidden_bias=np.random.uniform(0,1,size=(noofnodes,))       # random bais from range -1 to 1 
        hidden_op = np.random.uniform(0, 1, size=(noofnodes,1))   
        op_bias = np.random.uniform(0, 1, size=(1,)) 
        final.append(np.array(inp_hidden))
        final.append(np.array(hidden_bias))
        final.append(np.array(hidden_op))
        final.append(np.array(op_bias)) 
        classifier.set_weights(final)
            
    ##################################################
    ##################################################
    ##################################################
    
    import time
    stime = int(round(time.time() * 1000))
    
    final=classifier.get_weights()
    classifier.save_weights(nu+"Initial_Training_Assigned_ANNweight.h5")
    
    # Fitting the ANN to the Training set
    classifier.fit(train_in, train_out,batch_size = noofbatchsize, epochs =noofepoch,verbose=0)      
    
    import time
    etime = int(round(time.time() * 1000))
    
    # Predicting the Test set results
    y_pred = classifier.predict(test_in)
    y_pred = (y_pred > 0.5)
    #print(test_output.shape)
    cm=confusion_matrix(test_out, y_pred)
    #print(cm)
    effi=((cm[0][0]+cm[1][1])*100)/float(cm[0][0]+cm[1][1]+cm[1][0]+cm[0][1])
    precision=(cm[0][0]/float(cm[0][1]+cm[0][0]))
    recall=(cm[0][0]/float(cm[1][0]+cm[0][0]))
    f_measure=(2*precision*recall)/float(precision+recall)
    #print(effi)
    
    # serialize model to JSON
    #hiddennode_epoch_decay_learningrate
    OpenArray=classifier.get_weights()
    #print(OpenArray)
    
    classifier.save(nu+"full_model.h5")  # creates a HDF5 file 'my_model.h5'
    classifier.save_weights(nu+"After_Training_Final_ANNweight.h5")
    
    del classifier  # deletes the existing model
    
    # returns a compiled model
    # identical to the previous one
    #from keras.models import load_model
    #model = load_model('my_model.h5')
    
    '''
    model_json = classifier.to_json()
    with open(nu+"classifer.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    classifier.save_weights(nu+"model.h5")
    #print("Saved model to disk")
    '''
    
    
    return cm,effi,precision,recall,f_measure,OpenArray,final,stime,etime,train_in,test_in,final_train_in,corel_full_matrix
    

#----------------------------------------------------
#-----------------------MAIN-------------------------
#----------------------------------------------------
#----------------------------------------------------
#csvfiles=[]
#noofepoch=[]
#noofnodes=[]
#putmc=[]
#



# parameter  1 for Company Selection      #  1 for Reliance,  2 for Infosys  3 for SBI   4 for SunPharma
# parameter  2 for Model Selection        #  1 for Random weights, 2 for Pearson, 3 for Pearson absolute 
# parameter  3 for Starting Epoch
# parameter  4 for Ending Epoch
# parameter  5 for Starting Node
# parameter  6 for Ending Node
# parameter  7 for add_epoch_gap
# parameter  8 for putyear                #    2003 or 2008

import sys
gotParameters=sys.argv
cmpindex=int(gotParameters[1])   # 1 for Reliance 2 for Infosys
modelindex=int(gotParameters[2])    # 1 for random weights 2 for pearson 3 for pearson absolute 
putcmp=['','Reliance','Infosys','SBI','SunPharma','HDFC','DrReddy']   # Reliance, Infosy
putcmp_stockname=['','RELIANCEEQN','INFYEQN','SBINEQN','SUNPHARMAEQN','HDFCEQN','DRREDDYEQN']
putmodel=['','Random Weights','Pearson Weights','Pearson Weights ABSOLUTE']


Start_Epoch=int(gotParameters[3])
End_Epoch=int(gotParameters[4])
Start_Node=int(gotParameters[5])
End_Node=int(gotParameters[6])
add_epoch_gap=int(gotParameters[7])
putyear=int(gotParameters[8])

#----------------------------------------------------
#-----------------------MAIN CLOSE-------------------
#----------------------------------------------------
#----------------------------------------------------


gotcmp=putcmp[cmpindex]
gotmodel=putmodel[modelindex]
stockname=putcmp_stockname[cmpindex]

#for one_epoch in range(Start_Epoch,End_Epoch+1000,1000): 
#    for one_node in range(Start_Node,End_Node+10,10):
#        for one_mc in range(1,10,1):
#            one_mc=one_mc/10
#            csvfiles.append(stockname)
#            noofepoch.append(one_epoch)
#            noofnodes.append(one_node)
#            putmc.append(one_mc)
    

noofbatchsize=10 #BSZ BatchSize
noofsplitinratio=2 #SR SplitInRatio
putlr=0.1
putactivation='relu'   # tanh OR relu
putoptimizer='sgd'   # rmsprop OR adam OR custom (sgd Stochastic gradient descent )
putcustomoptimizer=True
putcustom=putcustomoptimizer

#----change the above parameters---------------------
#----------------------------------------------------
#----------------------------------------------------
#----------------------------------------------------

print("Models Started")
for one_epoch in range(Start_Epoch,End_Epoch+add_epoch_gap,add_epoch_gap): 
    for one_node in range(Start_Node,End_Node+10,10):  
        AllComputedEfficiency=[]
        AllComputedPrecision=[]
        AllComputedRecall=[]
        AllComputedFMeasure=[]
        AllTimeHH=[]
        AllTimeMM=[]
        AllTimeSS=[]
        AllTimeMS=[]
        AllTimeAllMS=[]
        for one_mc in range(1,10,1):
            one_mc=one_mc/10
            putfold_for_csv,putfoldername,putinnu=GiveFoldersAccordingToCustomChoice(putyear,gotcmp,gotmodel,stockname,putcustom,putactivation,putoptimizer,putlr,one_mc,noofsplitinratio,noofbatchsize,one_node,one_epoch)
            
            #print(putfoldername)
            path = putfoldername   # if folder doesn't exists then create new folder
            if not os.path.exists(path):
                os.makedirs(path)    
            allnu=putfoldername+'Runs'+putinnu
            nu=putfoldername+putinnu 
            
            
            #print(csvfiles[puti])
            train_in,test_in,train_out,test_out=get_dataset(putfold_for_csv,stockname,stockname,one_node,one_epoch,noofbatchsize,noofsplitinratio,putyear)
            train_in,test_in,train_out,test_out=save_data_mapping_with_dates(train_in,test_in,train_out,test_out)
            onecm,oneeffi,oneprecision,onerecall,onef_measure,final_ANNweight,Assigned_ANNweight,stime,etime,normalized_train_in,normalized_test_in,final_train_in,corel_full_matrix=compute_effi(modelindex,putfoldername,putcustom,putoptimizer,putactivation,putlr,one_mc,stockname,allnu,nu,train_in,test_in,train_out,test_out,one_node,one_epoch,noofbatchsize,noofsplitinratio)
            #print("\n\n"+csvfiles[puti])
            #print(onecm)
            #print("Efficiency : "+str(oneeffi))
            savemodel(stime,etime,allnu,nu,stockname,onecm,oneeffi,one_node,one_epoch,noofbatchsize,noofsplitinratio)
            final_ma=corel_full_matrix    
            final_ma.to_csv(putfoldername+'relation.csv', float_format='%.05f',sep=',',index=False) 
            putstr,puth,putm,puts,putmilli,putallmilli=get_run_time(stime,etime)
            #print("\n "+putstr)
            
            
            
            print("--> Completed -- Epoch "+str(one_epoch)+" Node "+str(one_node)+" MC "+str(one_mc*100))   
                
            
            AllComputedEfficiency.append(oneeffi)
            AllComputedPrecision.append(oneprecision)
            AllComputedRecall.append(onerecall)
            AllComputedFMeasure.append(onef_measure)
            AllTimeHH.append(puth)
            AllTimeMM.append(putm)
            AllTimeSS.append(puts)
            AllTimeMS.append(putmilli)
            AllTimeAllMS.append(putallmilli)
            
        #print("call save all")
        saveAllInCSV(putyear,gotcmp,gotmodel,one_node,one_epoch,stockname,AllComputedEfficiency,AllComputedPrecision,AllComputedRecall,AllComputedFMeasure,AllTimeHH,AllTimeMM,AllTimeSS,AllTimeMS,AllTimeAllMS)
        
print("Done-----------Done-------------Done")
print("--------------------------------------------------------------------------------------")
print("--------------------------------------------------------------------------------------")
print("----------------------------------"+gotcmp+"------------------------------------------")
print("----------------------------------"+gotmodel+"----------------------------------------")
print("-----------------Epoch--"+str(Start_Epoch)+" to "+str(End_Epoch)+"--------------------")
print("-----------------Node--"+str(Start_Node)+" to "+str(End_Node)+"-----------------------")
print("-----------------Year--------------"+str(putyear)+"-----------------------------------")
print("--------------------------------------------------------------------------------------")
print("--------------------------------------------------------------------------------------")
