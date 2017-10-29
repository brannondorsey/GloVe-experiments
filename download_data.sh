#!/bin/bash

curl -L http://www-nlp.stanford.edu/data/glove.6B.zip -o data/glove.6B.zip
unzip data/glove.6B.zip -d data/glove
rm data/glove.6B.zip
