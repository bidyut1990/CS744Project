from pyspark import SparkConf, SparkContext, SQLContext 

DIR = '/home/username/Downloads/'
dataDir = DIR+''
file = dataDir+'export.csv'
fileOut = dataDir+'sorted.csv'

conf = SparkConf().setMaster("local").setAppName("TableData")
sc = SQLContext(SparkContext(conf = conf))

data = sc.read.csv(file, header='true')
data.createGlobalTempView('Table')

sc.sql('''SELECT * FROM global_temp.Table ORDER BY cca2, timestamp''').coalesce(1).write.csv(fileOut)
