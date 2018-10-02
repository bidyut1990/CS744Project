#!/usr/bin/env python
# coding: utf-8

#import findspark
from operator import add
#findspark.init()
from pyspark import  SparkConf, SparkContext
from pyspark.sql import SparkSession

infile = "hdfs://128.104.222.61:9000/CS744/inputdata/enwiki-pages-articles/link-enwiki-20180601-pages-articles*"
fileOut = 'hdfs://128.104.222.61:9000/CS744/outputdata/outputranks_bigsample1.txt'

def flaten(tup):
    key,value = tup # (URL, (List of links, rank))
    links,rank = value
    n = len(links)
    out = [(dest,rank/n) for dest in links]
    #out.append((key, 0))# append the src also to this list, with no rank from this scenario
    return out

conf = SparkConf().setMaster("spark://128.104.222.61:7077").setAppName("TableData")
sc = SparkContext(conf = conf)

data = sc.textFile(infile)
#sample = data.zipWithIndex().filter(lambda x:x[1]<1000).map(lambda l:l[0])

links = data.map(lambda x: x.split('\t'))#break line
links = links.filter(lambda l: ':' not in l[1] or l[1][0:9] == 'Category:')#ignore some
links = links.map(lambda l: [k.lower() for k in l])#to lowercase
links = links.groupBy(lambda l: l[0])#group by key as from value
links = links.map(lambda x : (x[0], [l[1] for l in list(x[1])]))#convert iterator to list of values

#links.cache()

#N = links.count()
r0 = 1.0
iterations = 10

ranks = links.keys().map(lambda x: (x,r0))

for i in range(iterations):
    contribs = links.join(ranks).flatMap(flaten)
    ranks = contribs.reduceByKey(add).mapValues(lambda x: 0.15 + (0.85)*x)

#out = ranks.collect()
ranks.saveAsNewAPIHadoopFile(fileOut, "org.apache.hadoop.mapreduce.lib.output.TextOutputFormat","org.apache.hadoop.io.Text","org.apache.hadoop.io.Text")
