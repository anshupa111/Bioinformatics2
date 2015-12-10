# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 06:47:24 2015

@author: anshupa
"""
#url="https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls"


import pandas as pd
import sklearn.decomposition as skd
import matplotlib.pyplot as plt
#import pylab


########################################ASSIGNMENT 4 (a)#########################################
#mouse_data = pd.read_excel(url,sep='\t')
path="C:\\Users\\anshupa\\Downloads\\Data_Cortex_Nuclear.xls"
mouse_data = pd.read_excel(path,sep='\t')
classList=['t-CS-m','t-CS-s','c-CS-s','c-CS-m']
val=sum(mouse_data.isnull().values.ravel())
#print("Missing values count : ", val)
mod_mouse_data=mouse_data.interpolate(method="linear")
mod_mouse_data=mod_mouse_data.fillna(method='backfill')
val=sum(mod_mouse_data.isnull().values.ravel())
#print("Missing values count : ", val)
mousedata_tCSm = mod_mouse_data[mod_mouse_data["class"] =='t-CS-m']
mousedata_tCSs = mod_mouse_data[mod_mouse_data["class"] =='t-CS-s']
mousedata_cCSs = mod_mouse_data[mod_mouse_data["class"] =='c-CS-s']
mousedata_cCSm = mod_mouse_data[mod_mouse_data["class"] =='c-CS-m']
comboData=[mousedata_tCSs.iloc[:,:],mousedata_cCSs.iloc[:,:],mousedata_tCSm.iloc[:,:],mousedata_cCSm.iloc[:,:]]
mouseDataPCA=pd.concat(comboData)
#save the class information for class 't-CS-m','t-CS-s','c-CS-s' and 'c-CS-m'
classInfo=mouseDataPCA['class']
classInfo = classInfo.reset_index(drop=True)
#save the mouse information for class 't-CS-m','t-CS-s','c-CS-s' and 'c-CS-m'
mouseID=mouseDataPCA['MouseID']
mouseID = mouseID.reset_index(drop=True)
print("Number of instances : ",len(mouseDataPCA.index))
mouseDataPCA=mouseDataPCA.iloc[:,1:78]
#generate the explained variance ratio plot
xVal=list()
expVarianceRatio=list()
for i in range(1,11):
    pca = skd.PCA(i)
    tMouseData = pca.fit(mouseDataPCA).transform(mouseDataPCA)
    xVal.append(i)
    expVarianceRatio.append(sum(pca.explained_variance_ratio_))

plt.figure(figsize=(5, 5))
plt.plot(xVal,expVarianceRatio, linewidth=2)
plt.xticks(xVal)
plt.title('EXPLAINED VARIANCE RATIO', fontsize=14)
plt.xlabel('Principal component', fontsize=10)
plt.ylabel('Individual proportion of the covered variance', fontsize=10)
plt.show()
# n_components=8 
# Number of components need to cover ⿥ 95% of the variance is ⿥ 8
n_components=5
pca = skd.PCA(n_components)
#perform PCA on the mouse data
transformMouseData = pca.fit(mouseDataPCA).transform(mouseDataPCA)
transformMouseData=pd.DataFrame(transformMouseData)
print(sum(pca.explained_variance_ratio_))
explainedVariance=list(pca.explained_variance_ratio_)

######################################## ASSIGNMENT 4(b) ###########################################
### In the third principal component PC2 we see a clear effect of the treatment
#method to create scatter matrix
def make_scatterMatrix(transformMouseData,labelcs,labelcm,labelts,labeltm,principalComp):
    plt.figure(figsize=(20,10))
    for i in range(5):
        for j in range(5):
            if (i != j):
                plt.subplot2grid((len(principalComp),len(principalComp)),(i,j))
                plt.scatter(transformMouseData[i][labelcs],transformMouseData[j][labelcs],color = "#FF7F50",alpha = 0.8, label = 'c-CS-s')
            #plt.annotate(transformMouseData['MouseID'][labelcs],(transformMouseData[i][labelcs],transformMouseData[j][labelcs]))
                plt.scatter(transformMouseData[i][labelcm],transformMouseData[j][labelcm],color = "#90EE90",alpha = 0.8,label = 'c-CS-m')
                plt.scatter(transformMouseData[i][labelts],transformMouseData[j][labelts],color = "#B0E0E6",alpha = 0.8,label = 't-CS-s')
                plt.scatter(transformMouseData[i][labeltm],transformMouseData[j][labeltm],color = "#FFE4B5",alpha = 0.8,label = 't-CS-m')
            
            else:
                plt.subplot2grid((len(principalComp),len(principalComp)),(i,j))
                transformMouseData[i][labelcs].hist(color = "#FF7F50", alpha = 0.8,bins = 15, label = 'c-CS-s')
                transformMouseData[i][labelcm].hist(color = "#90EE90", alpha = 0.8,bins = 15, label = 'c-CS-m')
                transformMouseData[i][labelts].hist(color = "#B0E0E6", alpha = 0.8,bins = 15, label = 't-CS-s')
                transformMouseData[i][labeltm].hist(color = "#FFE4B5", alpha = 0.8,bins = 15, label = 't-CS-m')
                plt.title(principalComp[i])
            plt.legend(loc = 1, prop = {'size':7})
            plt.title(principalComp[i] + principalComp[j], fontsize = 12)
            plt.tight_layout()

    plt.show()

principalComp=transformMouseData.columns.values
#append class and mouse information to transformed data
transformMouseData['class']=classInfo
transformMouseData['MouseID']=mouseID
#divide the transformed data based on class
transformMouseData_tCSm = transformMouseData[transformMouseData["class"] =='t-CS-m']
transformMouseData_tCSs = transformMouseData[transformMouseData["class"] =='t-CS-s']
transformMouseData_cCSs = transformMouseData[transformMouseData["class"] =='c-CS-s']
transformMouseData_cCSm = transformMouseData[transformMouseData["class"] =='c-CS-m']
#extract the indices of the four classes
labelts = list(transformMouseData_tCSs.index)
labeltm = list(transformMouseData_tCSm.index)
labelcs = list(transformMouseData_cCSs.index)
labelcm = list(transformMouseData_cCSm.index)
#create a scatter matrix
make_scatterMatrix(transformMouseData,labelcs,labelcm,labelts,labeltm,principalComp)
########################################### ASSIGNMENT 4 (c) #########################################
#from scatter matrix obtained we get a rough idea that the outliers belong to class c-CS-s
#Annotate the scatterplot in which outliers are visible to get an idea of mouse which contains outliers
plt.figure(figsize=(20,10))
fig, ax = plt.subplots()
ax.scatter(transformMouseData[2][labelcs],transformMouseData[1][labelcs],color = "#FF7F50",alpha = 0.8, label = 'c-CS-s')
for i in labelcs:
    ax.annotate(transformMouseData['MouseID'][i],(transformMouseData[2][i],transformMouseData[1][i]))
plt.show()
#from annotation we get an idea that the outliers come from Mouse 3484 
# MOUSE IDs of outliers =[3484_1,3484_2,3484_3,3484_4,3484_5,3484_6,3484_7,3484_8,3484_9,3484_10,3484_11,3484_12,
#3484_13,3484_14,3484_15]
########################################## ASSIGNMENT 4 (d) ###########################################
####### After removing the outliers
#remove samples coming from 3484 outliers from transformed
mod_transformMouseData_cCSs=transformMouseData_cCSs[transformMouseData_cCSs['MouseID'].str.contains('3484')== False]
#mod_transformMouseData_cCSs=mod_transformMouseData_cCSs[mod_transformMouseData_cCSs['MouseID'].str.contains('3497')== False]
comboDataNew=[transformMouseData_tCSs.iloc[:,:],mod_transformMouseData_cCSs.iloc[:,:],transformMouseData_tCSm.iloc[:,:],transformMouseData_cCSm.iloc[:,:]]
transformMouseDataNew=pd.concat(comboDataNew)
labelcsNew = list(mod_transformMouseData_cCSs.index)
make_scatterMatrix(transformMouseDataNew,labelcsNew,labelcm,labelts,labeltm,principalComp)
#Uncomment the code for annotation
#annotate the plots to get an idea of clusters for by individual mouse
#plt.figure(figsize=(20,10))
#fig, ax = plt.subplots()
#ax.scatter(transformMouseDataNew[2][labelcsNew],transformMouseDataNew[1][labelcsNew],color = "#FF7F50",alpha = 0.8, label = 'c-CS-s')
#for i in labelcs:
    #ax.annotate(transformMouseDataNew['MouseID'][i],(transformMouseDataNew[2][i],transformMouseDataNew[1][i]))
#plt.show()
#name = "Assignment4"
#pylab.savefig(name + ".png")
mod_mouse_dataNew=mod_mouse_data
mouse_dataNew=mod_mouse_dataNew.iloc[:,1:78]
val=sum(mod_mouse_data.isnull().values.ravel())
#print("Missing values count : ", val)
mouse_data_cols=mouse_data.columns.values
mouseCols=list()
for i in mouse_data_cols:
    val=i.encode('ascii','ignore')
    mouseCols.append(val)
#print(mouseCols)
################################################ ASSIGNMENT 4(e) ###########################################
mod_mouse_dataNew=mod_mouse_data
mouse_dataNew=mod_mouse_dataNew.iloc[:,1:78]
val=sum(mod_mouse_data.isnull().values.ravel())
#print("Missing values count : ", val)
mouse_data_cols=mouse_data.columns.values
mouseCols=list()
for i in mouse_data_cols:
    val=i.encode('ascii','ignore')
    mouseCols.append(val)
#print(mouseCols)
mouse_data_proteins=mouseCols[1:78]
baseLineData=mod_mouse_data[(mod_mouse_data['class']== 'c-CS-s') & (mod_mouse_data['Genotype']=='Control')]
indexVal=mod_mouse_data.index
mean_BaseLine=list()
for i in mouse_data_proteins:
    meanVal=baseLineData[i].mean()
    mean_BaseLine.append(meanVal)

new_BaseLine=pd.DataFrame(mean_BaseLine)
new_BaseLine=new_BaseLine.transpose()
new_BaseLine.columns=mouse_dataNew.columns
#print(new_BaseLine)
newMouseDataPCA=mouse_dataNew.div(new_BaseLine.iloc[0])
#print(newMouseDataPCA)
n_components=20
#Number of components need to cover ⿥ 95% of the variance = 20
pca = skd.PCA(n_components)
pcaVal=pca.fit(newMouseDataPCA).transform(newMouseDataPCA)
print(sum(pca.explained_variance_ratio_))
