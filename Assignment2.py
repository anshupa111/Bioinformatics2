# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:16:21 2015

@author: anshupa
"""

######## Assignment-2 Part (a) ############
url="https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls"
import pandas as pd
import statistics as stat
mouse_data = pd.read_excel(url,sep='\t')
#mouse_data = pd.read_excel("C:\\Users\\anshupa\\Data_Cortex_Nuclear.xls",sep='\t')

print("Number of instances : ",len(mouse_data.index))
print("Number of columns : ",len(mouse_data.columns))
print("Column Names : ",mouse_data.columns.values)
mouse_data_cols=mouse_data.columns.values
mouse_data_proteins=mouse_data_cols[1:78]
val=sum(mouse_data.isnull().values.ravel())
print("Missing values count : ", val)
######## Assignment-2 Part (b) ############
#mod_mouse_data=mouse_data.interpolate(method="linear")
mod_mouse_data=mouse_data.fillna(value=-1)
value=sum(mod_mouse_data.isnull().values.ravel())
print("Missing values count : ", value)
######## Assignment-2 Part (c) ############
mousedata_tCSm = mod_mouse_data[mod_mouse_data["class"] =='t-CS-m']
print("Number of instances t-CS-m : ",len(mousedata_tCSm.index))
mousedata_tCSs = mod_mouse_data[mod_mouse_data["class"] =='t-CS-s']
print("Number of instances t-CS-s : ",len(mousedata_tCSs.index))
mousedata_cCSs = mod_mouse_data[mod_mouse_data["class"] =='c-CS-s']
print("Number of instances c-CS-s : ",len(mousedata_cCSs.index))
mousedata_cCSm = mod_mouse_data[mod_mouse_data["class"] =='c-CS-m']
print("Number of instances c-CS-m : ",len(mousedata_cCSm.index))
######## Assignment-2 Part (d) ############
#protein expression data for class t-CS-s
dataTCSs=mousedata_tCSs.iloc[:,1:78]
#print("Number of columns : ",len(dataTCSs.columns))
#protein expression data for class c-CS-s
dataCCSs=mousedata_cCSs.iloc[:,1:78]
#combining class t-CS-s and c-CS-s
combinedData=[dataTCSs,dataCCSs]
combTCSsCCSs=pd.concat(combinedData)
#compute F-Score
def computeFScore(protList,dataTCSs,dataCCSs,combTCSsCCSs):
    final_Prot={}
    topProteins={}
    for i in range(77):
        protein=protList[i]
        mean_TCSs=dataTCSs[protList[i]].mean()
        mean_CCSs=dataCCSs[protList[i]].mean()
        combo_mean=combTCSsCCSs[protList[i]].mean()
        numeratorFScore=(((mean_TCSs-combo_mean)**2)+(mean_CCSs-combo_mean)**2)
        denominatorFScore=((stat.variance(dataTCSs[protList[i]]))+(stat.variance(dataCCSs[protList[i]])))
        #denominatorFScore=((dataTCSs[protList[i]].var(ddof=True))+(dataCCSs[protList[i]].var(ddof=True)))
        FScore=numeratorFScore/denominatorFScore
        final_Prot[protein]=FScore
    
    sortedDict=sorted(final_Prot.items(), key=lambda x:x[1],reverse=True)
    topProteins=sortedDict[:5]
    return topProteins
    
#get list of top proteins
#as keys() function was not working for dictionary so have to use this approach
topProteinList={}
topProteinList=computeFScore(mouse_data_proteins,dataTCSs,dataCCSs,combTCSsCCSs)
proteinList=list()
fScoreList=list()
for i in range(5):
    protein,fscore=topProteinList[i]
    proteinList.append(protein)
    fScoreList.append(fScoreList)
print(proteinList)
######## Assignment-2 Part (e) ############
comboData=[mousedata_tCSs.iloc[:,1:78],mousedata_cCSs.iloc[:,1:78],mousedata_tCSm.iloc[:,1:78],mousedata_cCSm.iloc[:,1:78]]
finalData=pd.concat(comboData)
finalProteinData=finalData[proteinList]
print("Number of columns : ",len(finalProteinData.columns))



