# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 18:27:41 2018

@author: lenovo123
"""

import csv
import numpy as np
import pandas as pd

maxx = []
minn = []


############################
############################
############################

putyear=2008
cmpindex=1
cmplist=['','RELIANCEEQN','INFYEQN','SBINEQN','SUNPHARMAEQN','HDFCEQN','DRREDDYEQN']
putcmp=cmplist[cmpindex]

############################
############################
############################





for i in range(11):
    maxx.append(0);
    minn.append(100000000)

    
for yrs in range(putyear, putyear+10):
    lmin_col = []
    lmax_col = []

    with open('Dataset/computed_feature_01-01-'+str(yrs)+'-TO-31-12-'+str(yrs)+putcmp+'.csv', "r") as f_input:
        reader = csv.reader(f_input)
        next(reader)
        
        for row in reader:
            roww = []
            
            for col in row:
                if(any(c.isalpha() for c in col)):
                    continue
                else:
                    roww.append(float(col))
            
    
            if lmin_col:
                lmin_col = [min(x,y) for x,y in zip(lmin_col, roww)]
            else:
                lmin_col = roww

            if lmax_col:
                lmax_col = [max(x,y) for x,y in zip(lmax_col, roww)]
            else:
                lmax_col = roww
    
    lmin_col = np.array(lmin_col)
    lmax_col = np.array(lmax_col)
    
    
    for i in range(11):
        if(maxx[i] < lmax_col[i]):
            maxx[i] = lmax_col[i]
        if(minn[i] > lmin_col[i]):
            minn[i] = lmin_col[i]
    

c1 = []
c2 = []
c3 = []
c4 = []
c5 = []
c6 = []
c7 = []
c8 = []
c9 = []
c10 = []
c11 = []

rows = []




for yrs in range(putyear, putyear+10):
    with open('Dataset/computed_feature_01-01-'+str(yrs)+'-TO-31-12-'+str(yrs)+putcmp+'.csv', "r") as f_input:
        reader = csv.reader(f_input)
        next(reader)
        for row in reader:
            rows.append(row)
            
for i in range(len(rows)):
    c1.append(float(rows[i][1]))
    c2.append(float(rows[i][2]))
    c3.append(float(rows[i][3]))
    c4.append(float(rows[i][4]))
    c5.append(float(rows[i][5]))
    c6.append(float(rows[i][6]))
    c7.append(float(rows[i][7]))
    c8.append(float(rows[i][8]))
    c9.append(float(rows[i][9]))
    c10.append(float(rows[i][10]))
    c11.append(float(rows[i][11]))

mean = []
std = []

mean.append(np.mean(c1))
mean.append(np.mean(c2))
mean.append(np.mean(c3))
mean.append(np.mean(c4))
mean.append(np.mean(c5))
mean.append(np.mean(c6))
mean.append(np.mean(c7))
mean.append(np.mean(c8))
mean.append(np.mean(c9))
mean.append(np.mean(c10))
mean.append(np.mean(c11))

std.append(np.var(c1)**(1/2))
std.append(np.var(c2)**(1/2))
std.append(np.var(c3)**(1/2))
std.append(np.var(c4)**(1/2))
std.append(np.var(c5)**(1/2))
std.append(np.var(c6)**(1/2))
std.append(np.var(c7)**(1/2))
std.append(np.var(c8)**(1/2))
std.append(np.var(c9)**(1/2))
std.append(np.var(c10)**(1/2))
std.append(np.var(c11)**(1/2))


print("MAX ARRAY")
print(maxx)

print("MIN ARRAY")
print(minn)

print("MEAN")
print(mean)

print("STD DEV")
print(std)

final=[]
final.append(maxx)
final.append(minn)
final.append(mean)
final.append(std)

X=final  
X=np.array(X).T


#putindex=headSTR.split(' ')
        
df_mad= pd.DataFrame(X,columns=['Max','Min', 'Mean', 'StandardDeviation'])
df_mad.to_csv('got_MinMax_For_'+putcmp+'_'+str(putyear)+'.csv', sep=',') 
