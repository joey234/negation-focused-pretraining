#!/usr/bin/env sh

SRC=src

CORPUS=corpus
TR=$CORPUS/SEM-2012-SharedTask-PB-FOC-tr
DE=$CORPUS/SEM-2012-SharedTask-PB-FOC-de
TE=$CORPUS/SEM-2012-SharedTask-PB-FOC-te

WORDS=words/


echo "Merging training split ..." 
python $SRC/merge.py \
    $TR.txt \
    $TR.index \
    $WORDS../words/ \
    | $SRC/pretty_columns.py -i " " -o " " \
    > $CORPUS/merged/SEM-2012-SharedTask-PB-FOC-tr.merged

echo "Merging development split ..."
python $SRC/merge.py \
    $DE.txt \
    $DE.index \
    $WORDS../words/ \
    | $SRC/pretty_columns.py -i " " -o " " \
    > $CORPUS/merged/SEM-2012-SharedTask-PB-FOC-de.merged

echo "Merging test split ..."
python $SRC/merge.py \
    $TE.txt.rand \
    $TE.index.rand \
    $WORDS../words/ \
    | $SRC/pretty_columns.py -i " " -o " " \
    > $CORPUS/merged/SEM-2012-SharedTask-PB-FOC-te.merged

echo ""
echo "Done, files stored at $CORPUS/merged/"
