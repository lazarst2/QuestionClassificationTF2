import json
import sys
import os
import csv
from random import shuffle  






def main():
    data_dir = sys.argv[1]
    json_file = sys.argv[2]
    train_file = "train.tsv"
    test_file = "test.tsv"
    dev_file = "dev.tsv"

    full_train_file = "train_full.tsv"
    
    csv.register_dialect('myDialect', delimiter = '\t', skipinitialspace=True)
    
    
    lines = []
    infile = open(json_file,"r",encoding="utf8")

    fulltrainfile = open(os.path.join(data_dir,full_train_file),"w",encoding="utf8")

    writer = csv.writer(fulltrainfile, dialect='myDialect')
    
    for line in infile:
        lines.append(line)
        data = json.loads(line)
        d = []
        d.append(data["title"])
        d.append({"tags":data["tags"]})
        if len(data["tags"]) ==0:
            continue
        
        writer.writerow(d)
    infile.close()
    fulltrainfile.close()
    
    
    shuffle(lines)
    
    train_index = int(len(lines)*0.90)
    test_index = int(len(lines)*0.90)
    dev_index= len(lines)
    
    
    trainfile = open(os.path.join(data_dir,train_file),"w",encoding="utf8")
    
    writer = csv.writer(trainfile, dialect='myDialect')
    
    for line in lines[0:train_index]:
        data = json.loads(line)
        d = []
        d.append(data["title"])
        d.append(json.dumps({"tags":data["tags"]}))
        if len(data["tags"]) ==0:
            continue
        
        writer.writerow(d)
    
    
    trainfile.close()
    
    testfile = open(os.path.join(data_dir,test_file),"w",encoding="utf8")
    
    writer = csv.writer(testfile, dialect='myDialect')
    
    for line in lines[train_index:test_index]:
        data = json.loads(line)
        d = []
        d.append(data["title"])
        d.append(json.dumps({"tags":data["tags"]}))
        if len(data["tags"]) ==0:
            continue
        
        writer.writerow(d)
    
    
    testfile.close()
    
    devfile = open(os.path.join(data_dir,dev_file),"w",encoding="utf8")
    
    writer = csv.writer(devfile, dialect='myDialect')
    
    for line in lines[test_index:dev_index]:
        data = json.loads(line)
        d = []
        d.append(data["title"])
        d.append(json.dumps({"tags":data["tags"]}))
        if len(data["tags"]) ==0:
            continue
        
        writer.writerow(d)
    
    
    devfile.close()
    


if __name__=="__main__":
    main()