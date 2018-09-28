#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
from operator import add
findspark.init()
from pyspark import  SparkConf, SparkContext, SQLContext 
from pyspark.sql import SparkSession

#DIR = 'G:/edu/sem3/CS 744/'
#dataDir = DIR+'data/'
#file = dataDir+'web-BerkStan.txt'
infile = "hdfs://128.104.222.61:9000/users/Bidyut/web-BerkStan.txt"
fileOut = 'hdfs://128.104.222.61:9000/users/Bidyut/outputranks.txt'

def flaten(tup):
    _,t = tup # (URL, (List of links, rank))
    links,rank = t
    n = len(links)
    return [(dest,rank/n) for dest in links]

conf = SparkConf().setMaster("spark://128.104.222.61:7077").setAppName("TableData")
sc = SparkContext(conf = conf)

data = sc.textFile(infile)
#sample = data.zipWithIndex().filter(lambda x:x[1]<1000).map(lambda l:l[0])

links = data.filter(lambda x: x[0]!='#').map(lambda x: x.split('\t')).groupBy(lambda l: l[0]).map(lambda x : (x[0], [l[1] for l in list(x[1])]))
links.persist()

N = links.count()
r0 = 1.0
iterations = 10
a = 0.85

ranks = links.keys().map(lambda x: (x,r0))

for i in range(iterations):
    contribs = links.join(ranks).flatMap(flaten)
    #ranks = contribs.reduceByKey(add).mapValues(lambda x: a/N + (1-a)*x)
    #Tweaking the values acc. to  the assignment details
    ranks = contribs.reduceByKey(add).mapValues(lambda x: 0.15 + (0.85)*x)

out = ranks.collect()
sparkSession = SparkSession.builder.appName("pagerank").getOrCreate()
df = sparkSession.createDataFrame(out)
df.coalesce(1).write.csv(fileOut)

#f = open(fileOut, "w+")
#f.write(str(out))
#f.close()


#[l for l in out if l[0]=='1']

