#!/usr/bin/python

import sys
import os
import numpy as np
from nltk.tokenize import RegexpTokenizer
import mincemeat

def mapping(c, arr):
    for i in arr:
        yield i, [c, 1]

def reduce(c, arr):
    list = [0] * 67
    for task, i in arr:
        list[task] += i
    return list

dataPath = "./data/"
if len(sys.argv) == 2:
    dataPath = sys.argv[1]
fileNames = os.listdir(dataPath)

token = RegexpTokenizer(r'\w+')
data = []
for fileName in fileNames:
    with open(os.path.join(dataPath, fileName), 'r') as dataFile:
        data.append(token.tokenize(dataFile.read().lower()))

server = mincemeat.Server()
server.mapfn = mapping
server.reducefn = reduce
server.datasource = dict(enumerate(data))

results = server.run_server(password='changeme')
res = np.array(np.vstack(results.values()), dtype=np.int)
words = []
for key, _ in results.iteritems():
    words.append([key])
with open('Sherlock.csv', 'wb') as f:
    np.savetxt(f, np.vstack((np.array([""] + fileNames), np.hstack((np.array(words), res)))), delimiter=',', fmt='%4s')
