#!/usr/bin/env python

import sys

SECTIONS = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "23", "24"]

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "usage: merge.py FILE_ANNOTATIONS FILE_INDICES FOLDER_WORDS"
        sys.exit()

    file_annotations = open(sys.argv[1])
    file_indices     = open(sys.argv[2])    
    folder_words     = sys.argv[3]

    sections = {}
    for s in SECTIONS:
        sections[s] = []
        f = open(folder_words + "/%s.words" % s)
        sent = []
        for l in f.readlines():
            l = l.strip()   
            if l == "":
                sections[s].append(sent)
                sent = []
            else:
                sent.append(l)

    #for s in SECTIONS:
    #    print "read %s sentences from section %s" % (len(sections[s]), s)

    #print sections["13"][1872]

    words = []
    for l in file_indices.readlines():
        l = l.strip()

        section = l.split(" ")[-2]
        sentnum = int(l.split(" ")[-1])
        words += sections[section][sentnum]
        words += [""]
    
    for word, annotation in zip(words, file_annotations.readlines()):
        annotation = annotation.strip()
        print word, annotation
