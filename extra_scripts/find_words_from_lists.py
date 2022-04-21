#!/usr/bin/python
import argparse
import csv
import re
parser = argparse.ArgumentParser()                                               

# Sort a dictionary of word-frequency pairs in order of descending frequency.
def sortFreqDict(freqdict):
    aux = [(freqdict[key],key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

wordstring=''

parser.add_argument("--file", "-f", type=str, required=True)
parser.add_argument("--file_with_context", "-fc", type=str, required=True)
parser.add_argument("--topnwords", "-N", type=int, required=True)

args = parser.parse_args()

#remember this works only for lists after all...
with open (args.file, "r") as f:
    reader = csv.reader(f, delimiter="\t", quotechar=None)

    for i,line in enumerate(reader):
        if i<=args.topnwords: #find first N words
            #I want the second column which are the unique words for each category
            wordstring += line[1]+"\n"

wordlist = wordstring.split("\n")
#print (lines[0])
wordfreq = []
for word in wordlist:
    wordfrequency=0
    with open (args.file_with_context,"r") as f2:    
        for line in f2:
            if re.search(rf"\b{word}\b", line, re.IGNORECASE):
                wordfrequency+=1
    wordfreq.append(wordfrequency)

dictionary = dict(list(zip(wordlist,wordfreq)))
sorteddict = sortFreqDict(dictionary)

for s in sorteddict:
    #print only if found
    if (s[0] >0): 
        print(str(s[0])+"\t"+str(s[1]))