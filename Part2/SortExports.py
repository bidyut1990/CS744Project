from pyspark import SparkConf, SparkContext, SQLContext
import sys

inputfile = sys.argv[2]
outputfile = sys.argv[3]

masterip = sys.argv[1]
conf = SparkConf().setMaster(masterip).setAppName("TableData")
sc = SQLContext(SparkContext(conf = conf))

data = sc.read.csv(inputfile, header='true')
data.createGlobalTempView('Table')

sc.sql('''SELECT * FROM global_temp.Table ORDER BY cca2, timestamp''').coalesce(1).write.csv(outputfile)
