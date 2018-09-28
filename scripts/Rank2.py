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
file = dataDir+'link-enwiki-20180601-pages-articles14.xml-p7697599p7744799'
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


links = sample.map(lambda x: x.split('\t'))#break line
links = links.filter(lambda l: ':' not in l[1] or l[1][0:9] == 'Category:')#ignore some
links = links.map(lambda l: [k.lower() for k in l])#to lowercase
links = links.groupBy(lambda l: l[0])#group by key as from value
links = links.map(lambda x : (x[0], [l[1] for l in list(x[1])]))#convert iterator to list of values
links.persist()


# In[17]:


N = links.count()
r0 = 1
iterations = 10
a = 0.15


# In[18]:


ranks = links.keys().map(lambda x: (x,r0))


# In[19]:


for i in range(iterations):
    contribs = links.join(ranks).flatMap(flaten)
    ranks = contribs.reduceByKey(add).mapValues(lambda x: a/N + (1-a)*x)


# In[ ]:


#out = ranks.collect()


# In[ ]:


ranks.saveAsTextFile(fileOut)

