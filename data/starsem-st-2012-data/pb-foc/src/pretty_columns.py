#!/usr/bin/env python

import logging, sys, re
from optparse import OptionParser
import fileinput

def main(rows, delimiter):
    num_cols = 0
    for r in rows:
        if len(re.split("%s+" % delimiter, r)) > num_cols:
            num_cols = len(re.split("%s+" % delimiter, r))

    max = dict([(i, 0) for i in range(num_cols)])
    for r in rows:
        elems = re.split("%s+" % delimiter, r)
        for e, i in zip(elems, range(len(elems))):
            if len(e) > max[i]: max[i] = len(e)

    for r in rows:
        r = r.strip()
        elems = re.split("%s+" % delimiter, r)
        ## do not add blanks after the last column
        for e, i in zip(elems[:-1], range(len(elems)-1)):
            elems[i] = e + " " * (max[i]-len(e))
        yield elems

            
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    usage = "usage: %prog [options] FILES"
    parser = OptionParser(usage=usage)

    parser.add_option("-i", "--input_delimiter", default="\t",
                      help="Delimiter between columns in FILE")
    parser.add_option("-o", "--output_delimiter", default=" ",
                      help="Delimiter between columns in STDOUT")

    (options, args) = parser.parse_args()

    lines = [l for l in fileinput.input(files=args)]

    for r in main(lines, options.input_delimiter):
        print options.output_delimiter.join(r)

