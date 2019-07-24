import json
import sys
import os
import csv
from random import shuffle  






def main():
    data_dir = sys.argv[1]
    json_file = sys.argv[2]
    train_file = "train.json"
    test_file = "test.json"
    dev_file = "dev.json"

    full_train_file = "train_full.json"
    
    
    
    
    lines = []
    infile = open(json_file,"r",encoding="utf8")

    fulltrainfile = open(os.path.join(data_dir,full_train_file),"w",encoding="utf8")

    
    for line in infile:

        data = json.loads(line)
        d = []
        d.append(data["title"])
        d.append(data["tags"])
        if len(data["tags"]) ==0:
            continue

        lines.append(line)
        fulltrainfile.write(line)
    infile.close()
    fulltrainfile.close()
    
    
    shuffle(lines)
    
    train_index = int(len(lines)*0.90)
    test_index = int(len(lines)*0.90)
    dev_index= len(lines)
    
    
    trainfile = open(os.path.join(data_dir,train_file),"w",encoding="utf8")
    
    
    for line in lines[0:train_index]:
        
        trainfile.write(line)
    
    
    trainfile.close()
    
    testfile = open(os.path.join(data_dir,test_file),"w",encoding="utf8")
    
    
    for line in lines[train_index:test_index]:
        testfile.write(line)
    
    
    testfile.close()
    
    devfile = open(os.path.join(data_dir,dev_file),"w",encoding="utf8")
    
    
    for line in lines[test_index:dev_index]:
        devfile.write(line)
    
    
    devfile.close()
    


if __name__=="__main__":
    main()