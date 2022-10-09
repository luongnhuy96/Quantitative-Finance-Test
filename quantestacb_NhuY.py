#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import missingno as msno

#Read file data
df=pd.read_csv(r'C:\Users\QLRR\Downloads\rawdata.csv')
df


#1. How many rows and columns does the dataset contain
total_rows=len(df.axes[0]) #===> Axes of 0 is for a row
total_cols=len(df.axes[1]) #===> Axes of 1 is for a column
print("Number of Rows: "+str(total_rows))
print("Number of Columns: "+str(total_cols))




#2. Do you find any issue from this dataset
df.info()



print(df.isnull().sum())


#percentage of missing data
percent_missing = (df.isnull().sum().sort_values(ascending = False) * 100 / len(df)).round(2)
percent_missing



#Plot missing value
plt.figure(figsize=(10,6))
sns.displot(
    data=df.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)
plt.savefig("visualizing_missing_data_with_barplot_Seaborn_distplot.png", dpi=100)


# Plot amount of missingness
msno.bar(df)



#5.if missing value occur in the dataset, what method will you use to handle them

#->-Drop Rows with Missing Values with percentage of missing value <1% columns
#-> Apply machine learning kNN (k-Nearest Neighbor) for storm_name,continent_code


#6.how do you treat "date" column ?
#convert start_date to DateTime format
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
df



#which is the oldest date, the newest date and their range in days, month, quarters and years
list_of_dates= df['date']

print(min(list_of_dates)) # oldest date
print(max(list_of_dates)) # newest date


# In[752]:


df_dates = pd.DataFrame()
df_dates["datenew"]=pd.Series(list_of_dates)
df_dates 


# In[753]:


df_dates["datenew"] = pd.to_datetime(df_dates["datenew"])
# add a column for Year
df_dates['Year'] = df_dates["datenew"].dt.year
df_dates['Quarter'] = df_dates["datenew"].dt.quarter
df_dates['Month'] = df_dates["datenew"].dt.month
df_dates['Day'] = df_dates["datenew"].dt.day

df_dates


# In[754]:


#their range in days, month, quarters and years
print('range of day old:',min(df_dates['Day'])) # oldest day
print('range of day new:',max(df_dates['Day'])) # newest day

print('range of month old:',min(df_dates['Month'])) # oldest month
print('range of month new:',max(df_dates['Month'])) # newest month

print('range of quarter old:',min(df_dates['Quarter'])) # oldest quarter
print('range of quarter new:',max(df_dates['Quarter'])) # newest quarter


# In[755]:


print("newest Quarter of year:",df_dates.groupby(['Year'])['Quarter'].max())
print("smallest Quarter of year:",df_dates.groupby(['Year'])['Quarter'].min())


# In[756]:


print("newest month of year:",df_dates.groupby(['Year'])['Month'].max())
print("smallest month of year:",df_dates.groupby(['Year'])['Month'].min())


# In[757]:


print("newest day of month:",df_dates.groupby(['Month'])['Day'].max())
print("smallest day of month:",df_dates.groupby(['Month'])['Day'].min())


# In[758]:


date1 = pd.Timestamp(datetime.now()) #CURRENT_DATE
date2 = max(list_of_dates)#NEWEST_DATE

dt_day = date1.to_period('D') - date2.to_period('D')
print(dt_day )


# In[759]:


dt_months = date1.to_period('M') - date2.to_period('M')
dt_months


# In[760]:


dt_quarter = date1.to_period('Q') - date2.to_period('Q')
dt_quarter


# In[761]:


dt_years = date1.to_period('Y') - date2.to_period('Y')
dt_years


# In[762]:


#Do you find any invalid GPS coordinate ("geolocation") in the given dataset
#given latitude must be within [-90,90] and longitude must with [-180,180]
pd.set_option('display.max_columns', None,'display.max_rows', None)
df


# In[763]:


df['latitude']=''
df['longitude']=''
for i in range (0,len(df)):
    df['latitude'][i]=df['geolocation'][i].split(',', 1)[0].split('(')[1]
    df['longitude'][i]=df['geolocation'][i].split(',', 1)[1].split(')')[0]


# In[764]:


df


# In[765]:


df_geo = pd.DataFrame(data=df[['latitude','longitude']]).astype(float)
df_geo


# In[766]:


# invalide latitude out range [-90,90]
invalid_latitude_df=df_geo[(df_geo['latitude'] <(-90))| (df_geo['latitude'] >90)]
invalid_latitude_df


# In[767]:


# invalide longitude out range [-180,180]
invalid_longitude_df=df_geo[(df_geo['longitude'] <(-180))| (df_geo['longitude'] >180)]
invalid_longitude_df


# In[768]:


df_geo.isnull().sum()


# In[769]:


print("max latitude:",max(df_geo['latitude']))
print("min latitude:",min(df_geo['latitude']))

print("max longitude:",max(df_geo['longitude']))
print("min longitude:",min(df_geo['longitude']))


# In[770]:


df_geo['num_bins_latitude'] =pd.cut(x = df_geo['latitude'],
                        bins = [-100,-90,90],
                        labels = ["invalid", " valid"])
df_geo['num_bins_longitude'] =pd.cut(x = df_geo['longitude'],
                        bins = [-180,180,200],
                        labels = [" valid", "invalid"])
df_geo


# In[ ]:





# In[771]:


df_geo['label']=''
df_geo.loc[(df_geo['num_bins_latitude']=='invalid') | (df_geo['num_bins_longitude'] =='invalid'),'label']='invalid'
df_geo.loc[(df_geo['num_bins_latitude']!='invalid') & (df_geo['num_bins_longitude'] !='invalid'),'label']='valid'


# In[772]:


df_geo


# In[773]:


df_geo[(df_geo['num_bins_latitude']=='invalid')  | (df_geo['num_bins_longitude'] =='invalid')]


# In[774]:


df_geo2=df_geo[['latitude','longitude','label']]
df_geo2


# In[775]:


df_geo[(df_geo['num_bins_latitude']!='invalid')  | (df_geo['num_bins_longitude'] !='invalid')]


# In[776]:


df_geo['num_bins_latitude'] .value_counts()


# In[777]:


df_geo['num_bins_longitude'] .value_counts()


# In[778]:


sns.boxplot(x=df_geo['latitude'],data=df_geo)


# In[779]:


sns.boxplot(x=df_geo['longitude'],data=df_geo)


# In[ ]:





# In[780]:


#Using Scatter Plot
fig, ax = plt.subplots()
groups = df_geo2.groupby('label')
for name, group in groups:
    ax.plot(df_geo2.longitude, df_geo2.latitude, marker='o', linestyle='', markersize = 5, 
label=name)
ax.legend(numpoints=1)
#ax.set_ylim((-200, 200))


# In[781]:


df.info()


# In[782]:


df.describe()[['distance','population']]


# In[783]:


df


# In[784]:


#do you find any outlier regarding "population" and "distance"

# Scatter plot
fig, ax = plt.subplots(figsize = (18,10))
ax.scatter(df['distance'], df['population'])
 
# x-axis label
ax.set_xlabel('distance')
 
# y-axis label
ax.set_ylabel('population')
plt.show()


# In[785]:


# Position of the Outlier
print(np.where((df['population']>0.4) & (df['distance']>30)))


# In[786]:


#Using Boxplot
import seaborn as sns
sns.boxplot(df['distance'])


# In[787]:


# Position of the Outlier
print(np.where(df['distance']>20))


# In[788]:


import seaborn as sns
#define figure size
sns.set(rc={"figure.figsize":(6, 130)}) #width=8, height=4
sns.boxplot(df['population'])


# In[789]:


print(np.where(df['population']>0.2))


# In[790]:


#Using histogram
#import library
import seaborn as sns

#Using distplot function, create a graph
sns.distplot( a=df['distance'], hist=True)


# In[791]:


mean=df['distance'].mean()
std=df['distance'].std()
threshold = 3
outlier = [] 
for i in df['distance']: 
    z = (i-mean)/std 
    if z > threshold: 
        outlier.append(i) 
print('outlier in dataset is', outlier) 


# In[792]:


#import library
import seaborn as sns
#Iris Dataset
#Using distplot function, create a graph
sns.distplot( a=df['population'], hist=True)


# In[793]:


mean=df['population'].mean()
std=df['population'].std()
threshold = 3
outlier = [] 
for i in df['population']: 
    z = (i-mean)/std 
    if z > threshold: 
        outlier.append(i) 
print('outlier in dataset is', outlier) 


# In[794]:


df


# In[ ]:





# In[806]:


import pandas as pd
import plotly.express as px


fig = px.scatter_geo(df, lat='latitude', lon='longitude',hover_name="country_name",color = "country_name")

fig.show()


# In[808]:


#=>latitude=36.0186, longitude=180.7469
#=>latitude=-90.56401, longitude=-84.4086
# SPOT OUT THE POINT
df3 = pd.DataFrame(dict(lat=[-90.56401, 36.0186], lon=[-84.4086, 180.7469], subreg=[62, 93]))
fig = px.scatter_geo(df3, lat="lat", lon="lon", color="subreg")
fig.update_traces(marker=dict(size=50))
fig.show()



df_geo[(df_geo['num_bins_latitude']=='invalid')  | (df_geo['num_bins_longitude'] =='invalid')]

#output (containing id, date, country name, and geolocation of the points)

datageo= pd.DataFrame(data=df)
datageo['latitude']=datageo['latitude'].astype(float)
datageo['longitude']=datageo['longitude'].astype(float)
datageo

datageo['num_bins_latitude'] =pd.cut(x = datageo['latitude'],
                        bins = [-100,-90,90],
                        labels = ["invalid", " valid"])
datageo['num_bins_longitude'] =pd.cut(x = datageo['longitude'],
                        bins = [-180,180,200],
                        labels = [" valid", "invalid"])
datageo



datageo['label']=''
datageo.loc[(datageo['num_bins_latitude']=='invalid') | (datageo['num_bins_longitude'] =='invalid'),'label']='invalid'
datageo.loc[(datageo['num_bins_latitude']!='invalid') & (datageo['num_bins_longitude'] !='invalid'),'label']='valid'



datageo['label']=''
datageo.loc[(datageo['num_bins_latitude']=='invalid') | (datageo['num_bins_longitude'] =='invalid'),'label']='invalid'
datageo.loc[(datageo['num_bins_latitude']!='invalid') & (datageo['num_bins_longitude'] !='invalid'),'label']='valid'


datageo



datageo[(datageo['num_bins_latitude']=='invalid')  | (datageo['num_bins_longitude'] =='invalid')][['id','date','country_name','geolocation']]

