#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue March 31 22:39:54 2020

@author: yaoyao
"""
import json
import os, stat
from pprint import pprint
Dataset={}
Datascore={}
retval = os.getcwd()
#print("当前工作目录为  ：" + retval)
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
result=sorted(Datascore.items(), lambda x, y: cmp(x[1], y[1]))
        
        