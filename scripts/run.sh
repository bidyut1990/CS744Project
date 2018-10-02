echo $#
if [ "$#" -ne 3 ];then 
	echo "Please provide in this format: ./run.sh <spark master IP:Port> <Input file location> <output file location>"
	exit
fi
spark-submit --master $1 PageRank.py $1 $2 $3 
