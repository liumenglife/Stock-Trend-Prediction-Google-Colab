# -*- coding: utf-8 -*-
"""
Created on Tue May  1 18:32:44 2018

@author: Dhaval
"""

import numpy as np
import os
import pandas as pd




def compute_all_results(mode,noofepoch,noofnode,fold):
    All_final=0
    for i in range(len(noofepoch)):        
        print(noofnode[i])
        if(mode==0):
            putpath=fold+'\Results\Epoch_'+str(noofepoch[i])+'\\Nodes_'+str(noofnode[0])+'\\Nodes_'+str(noofnode[0])+'_Epoch_'+str(noofepoch[i])+'_All_Efficiency_Of_Training.csv'
        elif(mode==1):
            putpath=fold+'\Results\Epoch_'+str(noofepoch[i])+'\\Nodes_'+str(noofnode[0])+'\\Nodes_'+str(noofnode[0])+'_Epoch_'+str(noofepoch[i])+'_All_Efficiency_Of_Testing.csv'
        
        print(putpath)
        df=pd.read_csv(putpath)
        #print(df)
        final=df.iloc[0:1,:]
        for j in range(1,10):
            
            if(mode==0):
                putpath=fold+'\Results\Epoch_'+str(noofepoch[i])+'\\Nodes_'+str(noofnode[j])+'\\Nodes_'+str(noofnode[j])+'_Epoch_'+str(noofepoch[i])+'_All_Efficiency_Of_Training.csv'
            elif(mode==1):
                putpath=fold+'\Results\Epoch_'+str(noofepoch[i])+'\\Nodes_'+str(noofnode[j])+'\\Nodes_'+str(noofnode[j])+'_Epoch_'+str(noofepoch[i])+'_All_Efficiency_Of_Testing.csv'
            
            print(putpath)
            df=pd.read_csv(putpath)
            df=df.iloc[0:1,:]
            final = pd.concat([final, df], axis=0)
        final=final.reset_index()
        final=final.iloc[:,1:]   
        
        if(mode==0):
            path = fold+'/All_Results/Training'   # if folder doesn't exists then create new folder
        elif(mode==1):
            path = fold+'/All_Results/Testing'   # if folder doesn't exists then create new folder
           
        if not os.path.exists(path):
            os.makedirs(path)    
        
        if(mode==0):
            final.to_csv(path+'/ALL_Results_for_Epoch_'+str(noofepoch[i])+'.csv', sep=',',index=False) 
        elif(mode==1):
            final.to_csv(path+'/ALL_Results_for_Epoch_'+str(noofepoch[i])+'.csv', sep=',',index=False) 
        

        if(i==0):  # for first initiallization
            All_final=final
        else:
            All_final = pd.concat([All_final, final], axis=0)
        
    if(mode==0):
        All_final.to_csv(path+'/ALL_Combined_Results_Of_Training.csv', sep=',',index=False) 
    elif(mode==1):
        All_final.to_csv(path+'/ALL_Combined_Results_Of_Testing.csv', sep=',',index=False) 
    
    result=np.array(All_final)
    return result

#-------------------------------------------
#-------------------------------------------
#-------------------------------------------
#-------------------------------------------
#-------------------------------------------
#-------------------------------------------

noofepoch=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
#noofepoch=[1000,2000,3000,4000,5000,6000,9000]

getTop=5
mode=1    # MODE 0 FOR Training # MODE 1 FOR Testing #  MODE 2 FOR AVG   
putnode=10
modelindex=3    # 1 for random weights 2 for pearson 3 for pearson absolute 
cmpindex=1
putcmp=['','Reliance','Infosys','SBI','SunPharma']   # Reliance, Infosys
putmodel=['','Random Weights','Pearson Weights','Pearson Weights ABSOLUTE']


#-------------------------------------------
#-------------------------------------------
#-------------------------------------------
#-------------------------------------------
#-------------------------------------------
#-------------------------------------------


putfold=".\\"+putcmp[cmpindex]+" 20\\"+putmodel[modelindex]
noofnode=[]
   

for i in range(10):
    noofnode.append(putnode)
    putnode+=10

          
if(mode==2):
    result0=compute_all_results(0,noofepoch,noofnode,putfold)
    result1=compute_all_results(1,noofepoch,noofnode,putfold)
    result=np.add(result0,result1)
    result=result/2
else:
    result=compute_all_results(mode,noofepoch,noofnode,putfold)
    
print(result.shape)
putresult=np.reshape(result, (1, result.shape[0]*result.shape[1]))

print(putresult.shape)
x=putresult[0].tolist()
z=np.argsort(x)[(-1*getTop):][::-1]
print(z)




if(mode==0):
    saveFile = open(putfold+'/All_Results/Top_'+str(getTop)+'_Of_Training.txt','w')
elif(mode==1):
    saveFile = open(putfold+'/All_Results/Top_'+str(getTop)+'_Of_Testing.txt','w')
else:
    saveFile = open(putfold+'/All_Results/Top_'+str(getTop)+'_Of_Both.txt','w')
    
print("Epoch :  Node : momentum constant")

saveFile.write("\n\n "+putcmp[cmpindex]+" "+putmodel[modelindex]+" as Weights")    
saveFile.write("\n\nEpoch :  Node : momentum constant")    
   
ans=[]

for i in range(getTop):
    #xi=int((z[i]+1)/9)-1
    #xj=((z[i]+1)%9)-1
    xi=int((z[i])/9)
    xj=((z[i])%9)
    epochx=int(xi/10)
    nodex=int(xi%10)
    putstr=str(noofepoch[epochx])+" : "+str(noofnode[nodex])+"  : "+str((xj+1)/10)
    
    f=[]
    #f.append(putstr)
    #putnum='%.3f' % result[xi][xj]
    putnum=str(result[xi][xj])
    #f.append(noofepoch[epochx])
    #f.append(noofnode[nodex])
    #f.append(str((xj+1)/10))
    putstr="  "+str(noofepoch[epochx])+" : "+str(noofnode[nodex])+"  : "+str((xj+1)/10)
    f.append(putstr)
    f.append(putnum)
    ans.append(f)
    print(putstr)
    print(result[xi][xj])
    saveFile.write("\n\n"+putstr+"\n\t")    
    saveFile.write(str(result[xi][xj]))
    
saveFile.close()

#print(ans)

ans=pd.DataFrame(ans)

if(mode==0):
    ans.to_csv(putfold+'/All_Results/Top_'+str(getTop)+'_Of_Training.csv', sep=',',index=False) 
elif(mode==1):
    ans.to_csv(putfold+'/All_Results/Top_'+str(getTop)+'_Of_Testing.csv', sep=',',index=False) 
else:
    ans.to_csv(putfold+'/All_Results/Top_'+str(getTop)+'_Of_Both.csv', sep=',',index=False) 
    