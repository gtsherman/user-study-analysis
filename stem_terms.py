import sys

from nltk.stem import SnowballStemmer


stemmer = SnowballStemmer('english')
with open(sys.argv[1]) as f:
  for line in f:
    user, doc, index, term = line.strip().split(',')
    print(','.join([user, doc, index, stemmer.stem(term)]))
