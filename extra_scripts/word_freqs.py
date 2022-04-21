#!/usr/bin/python
import csv
import argparse
parser = argparse.ArgumentParser()                                               

#if I want to count per word from collections import Counter is much faster

# Given a list of words, return a dictionary of word-frequency pairs.
def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))


# Sort a dictionary of word-frequency pairs in order of descending frequency.

def sortFreqDict(freqdict):
    aux = [(freqdict[key],key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

wordstring=''

parser.add_argument("--file", "-f", type=str, required=True)
args = parser.parse_args()

#remember this works only for lists after all...
with open (args.file, "r") as f:
    reader = csv.reader(f, delimiter="\t", quotechar=None)
    for line in reader:
        #print (line)
        wordstring += line[0].lower()+"\t"

wordlist = wordstring.split("\t")
#print (lines[0])

dictionary = wordListToFreqDict(wordlist)
sorteddict = sortFreqDict(dictionary)

for s in sorteddict: 
    print(str(s[0])+"\t"+str(s[1])) 