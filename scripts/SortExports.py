#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
import pandas as pd
findspark.init()
from pyspark import SparkConf, SparkContext, SQLContext 


# In[2]:


DIR = 'G:/edu/sem3/CS 744/'
dataDir = DIR+'data/'
file = dataDir+'export.csv'
fileOut = dataDir+'sorted.csv'


# In[3]:



conf = SparkConf().setMaster("local").setAppName("TableData")
sc = SQLContext(SparkContext(conf = conf))


# In[4]:


data = sc.read.csv(file, header='true')


# In[5]:


data.createGlobalTempView('Table')


# In[6]:


#sc.sql('''SELECT * FROM global_temp.Table ORDER BY cca2, timestamp''').write.csv("fileOut.csv")


# In[7]:


data.write.csv(fileOut)

