# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:55:47 2018

@author: Dhaval
"""

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
 




def saveAllInCSV(node,epoch,EfficiencyToSave,PrecisionToSave,RecallToSave,FMeasureToSave,LenForMC):
    EfficiencyToSave=np.array(EfficiencyToSave)
    PrecisionToSave=np.array(PrecisionToSave)
    RecallToSave=np.array(RecallToSave)
    FMeasureToSave=np.array(FMeasureToSave)
    
    FinalSave=[]
    for i in range(LenForMC):
        temp=[]
        temp.append(EfficiencyToSave[i])
        temp.append(PrecisionToSave[i])
        temp.append(RecallToSave[i])
        temp.append(FMeasureToSave[i])
        FinalSave.append(temp)    
    FinalSave=np.array(FinalSave)
    FinalSave=FinalSave.transpose()
    
    
    path = 'Results/Epoch_'+str(epoch)+'/Nodes_'+str(node)+'/'   # if folder doesn't exists then create new folder
    if not os.path.exists(path):
        os.makedirs(path)  
    np.savetxt(path+'Nodes_'+str(node)+'_Epoch_'+str(epoch)+'_All_Efficiency_Of_Training.csv',FinalSave, fmt='%.10f',delimiter=',', header='Momentum01,02,03,04,05,06,07,08,09')
    
    

def savemodel(allnu,nu,onecm,oneeffi,noofnodes,noofepoch,noofbatchsize,noofsplitinratio):    
    saveFile = open(allnu+'_Of_Training.txt','w')
    saveFile.write("\n"+nu)    
    saveFile.write("\n\t")
    saveFile.write(str(onecm[0])+"\n\t")
    saveFile.write(str(onecm[1]))
    saveFile.write("\n\n")
    saveFile.write("\n\t\tEfficiency : "+str(oneeffi)+"\n")
    saveFile.write("\n\n")
    saveFile.write("\n\n--------------------------------------------------------------------------\n\n")
    saveFile.close()


def GiveFoldersAccordingToCustomChoice(putcustom,putactivation,putoptimizer,putlr,putmc,noofsplitinratio,noofbatchsize,noofnodes,noofepoch):
    
    # NodesAndEpochs  or   OptimizerMethod   or Company  
    p2='Details_Epoch'+str(noofepoch)+'/Node'+str(noofnodes)+'/MC'+str(putmc*100)  
    #p2='Updated_Node'+str(noofnodes)+'_Epoch'+str(noofepoch)  
    p3='ANN'
    
    if(putcustom):
        p3=p3+'_C'
        if(putoptimizer=='sgd'):
            p3=p3+'_SGD'
            putinnu='_LR'+str(int(putlr*100))+'MC'+str(int(putmc*100))+'SR1to'+str(noofsplitinratio)+'_BSZ'+str(noofbatchsize)
            putfoldername='./'+p3+'/'+p2+'/'
        elif(putoptimizer=='adam'):
            p3=p3+'_Adam'
            putinnu='_LR'+str(int(putlr*100))+'MC_NotDefine_'+'SR1to'+str(noofsplitinratio)+'_BSZ'+str(noofbatchsize)
            putfoldername='./'+p3+'/'+p2+'/' 
    else:
        p3=p3+'_default'
        p3=p3+'_'+putoptimizer
        if(putoptimizer=='adam'):
            putinnu='_LR_0_001'+'MC_NotDefine_'+'SR1to'+str(noofsplitinratio)+'_BSZ'+str(noofbatchsize)
        else:    
            putinnu='_LR_NotDefine'+'MC_NotDefine_'+'SR1to'+str(noofsplitinratio)+'_BSZ'+str(noofbatchsize)
        putfoldername='./'+p3+'/'+p2+'/' 

    return putfoldername,putinnu
	

def compute_effi(putfoldername,putcustom,putoptimizer,putactivation,putlr,putmc,allnu,nu,noofnodes,noofepoch,noofbatchsize,noofsplitinratio):
    
    '''
    # Model reconstruction from JSON file
    with open(nu+"classifer.json",'r') as f:
        model = model_from_json(f.read())
    
    # Load weights into the new model
    model.load_weights(nu+"model.h5")
    '''    
    from keras.models import load_model
    model = load_model(nu+"model.h5")
    
    
    #print("do")
    final_train_in=pd.read_csv(putfoldername+'Data_TrainingInput.csv')
    #print(final_train_in.shape)
    
    train_out=pd.read_csv(putfoldername+'Data_TrainingOutput.csv')    
    train_in=final_train_in.iloc[:,0:10]
    #print(train_in.shape)
    
    # Predicting the Test set results
    y_pred = model.predict(train_in)
    y_pred = (y_pred > 0.5)
    cm=confusion_matrix(train_out, y_pred)
    
    #print(cm)
    effi=((cm[0][0]+cm[1][1])*100)/float(cm[0][0]+cm[1][1]+cm[1][0]+cm[0][1])
    precision=(cm[0][0]/float(cm[0][1]+cm[0][0]))
    recall=(cm[0][0]/float(cm[1][0]+cm[0][0]))
    f_measure=(2*precision*recall)/float(precision+recall)
    #print(effi)
    
    return cm,effi,precision,recall,f_measure
    

#----------------------------------------------------
#-----------------------MAIN-----------------------
#----------------------------------------------------
#----------------------------------------------------



csvfiles=[]
noofepoch=[]
noofnodes=[]
putmc=[]
addmc=0.1

addEpoch=2
addNodes=10
addmc=0.1
LenForMC=9
LenForNode=2


for j in range(LenForNode):
    addmc=0.1
    for i in range(LenForMC):
        noofepoch.append(addEpoch)
        noofnodes.append(addNodes)
        putmc.append(addmc)
        addmc=addmc+0.1
    addNodes+=10

howmanycompanies=len(csvfiles)
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

for ii in range(LenForNode):   
    AllComputedEfficiency=[]
    AllComputedPrecision=[]
    AllComputedRecall=[]
    AllComputedFMeasure=[]

    for i in range(LenForMC):
        puti=(ii*9)+i
        putfoldername,putinnu=GiveFoldersAccordingToCustomChoice(putcustom,putactivation,putoptimizer,putlr,putmc[puti],noofsplitinratio,noofbatchsize,noofnodes[puti],noofepoch[puti])
        
        print(putfoldername)
        
        path = putfoldername   # if folder doesn't exists then create new folder
        if not os.path.exists(path):
            os.makedirs(path)    
        allnu=putfoldername+'Runs'+putinnu
        nu=putfoldername+putinnu 
        
        #print(csvfiles[puti])
        onecm,oneeffi,oneprecision,onerecall,onef_measure=compute_effi(putfoldername,putcustom,putoptimizer,putactivation,putlr,putmc[puti],allnu,nu,noofnodes[puti],noofepoch[puti],noofbatchsize,noofsplitinratio)
        #print("\n\n"+csvfiles[puti])
        #print(onecm)
        print("Efficiency : "+str(oneeffi))
        
        savemodel(allnu,nu,onecm,oneeffi,noofnodes[puti],noofepoch[puti],noofbatchsize,noofsplitinratio)
        #print("model saved")
        
        AllComputedEfficiency.append(oneeffi)
        AllComputedPrecision.append(oneprecision)
        AllComputedRecall.append(onerecall)
        AllComputedFMeasure.append(onef_measure)
           
        if(i>7):
            import webbrowser
            url = "https://www.google.com/search?q=epoch "+str(noofepoch[puti])+" node "+str(noofnodes[puti])+" mc "+str(putmc[puti]*100)+""   
            #url="file:///D:/mini%20project/Cool/Remove%20AD%20Oc%20Final%20RUNS/runurl.html?q=epoch "+str(noofepoch[puti])+" node "+str(noofnodes[puti])+" mc "+str(putmc[puti]*100)+""
            webbrowser.open(url)
            #print(url)
        
    
    saveAllInCSV(noofnodes[ii*9],noofepoch[ii*9],AllComputedEfficiency,AllComputedPrecision,AllComputedRecall,AllComputedFMeasure,LenForMC) 
