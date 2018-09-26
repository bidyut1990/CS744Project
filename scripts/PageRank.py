#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
from operator import add
findspark.init()
from pyspark import SparkConf, SparkContext, SQLContext 


# In[2]:


DIR = 'G:/edu/sem3/CS 744/'
dataDir = DIR+'data/'
file = dataDir+'web-BerkStan.txt'
fileOut = dataDir+'ranks.txt'


# In[3]:


def flaten(tup):
    _,t = tup # (URL, (List of links, rank))
    links,rank = t
    n = len(links)
    return [(dest,rank/n) for dest in links]


# In[4]:


conf = SparkConf().setMaster("local").setAppName("TableData")
sc = SparkContext(conf = conf)


# In[5]:


data = sc.textFile(file)


# In[6]:


sample = data.zipWithIndex().filter(lambda x:x[1]<1000).map(lambda l:l[0])


# In[7]:


links = sample.filter(lambda x: x[0]!='#').map(lambda x: x.split('\t')).groupBy(lambda l: l[0]).map(lambda x : (x[0], [l[1] for l in list(x[1])]))
links.persist()


# In[8]:


N = links.count()
r0 = 0.25
iterations = 10
a = 0.25


# In[9]:


ranks = links.keys().map(lambda x: (x,r0))


# In[10]:


for i in range(iterations):
    contribs = links.join(ranks).flatMap(flaten)
    ranks = contribs.reduceByKey(add).mapValues(lambda x: a/N + (1-a)*x)


# In[11]:


out = ranks.collect()


# In[13]:


[l for l in out if l[0]=='1']

