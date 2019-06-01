import sys

from nltk.stem import SnowballStemmer


stemmer = SnowballStemmer('english')
with open(sys.argv[1]) as f:
  for line in f:
    doc, term, weight = line.strip().split(',')
    print(','.join([doc, stemmer.stem(term), weight]))
