#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue March 31 22:39:54 2020

@author: yaoyao
"""
import json
import csv
import os, stat
import sys
import codecs
from pprint import pprint
Dataset={}
Datascore={}
retval = os.getcwd()
#print("current working folder  ：" + retval)
#os.chdir ('/Users/yaoyao/Desktop/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv')
#os.chmod('test1.txt', stat.S_IROTH)
index=0
domain = os.path.abspath(r'/Users/yaoyao/Desktop/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv')
for info in os.listdir(r'/Users/yaoyao/Desktop/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv'):
    info = os.path.join(domain,info)
    with open(info, 'r') as f:
        data = json.load(f)
        pprint(data["metadata"])
        Dataset[index]=data
        index+=1
        Datascore[index]=rank(data)# put the search engine result here for each file
        with open(str(index)+'.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_ALL)
            flag = True
            for line in f:
                dic = json.loads(line[0:-1])
                if flag:
                    keys = list(dic.keys())
                    writer.writerow(keys) 
                    flag = False
                writer.writerow(list(dic.values()))
        csvfile.close()
    f.close()
result=sorted(Datascore.items(), lambda x, y: cmp(x[1], y[1]))
        
        
