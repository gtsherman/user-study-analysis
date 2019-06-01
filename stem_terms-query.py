import sys

from nltk.stem import SnowballStemmer


stemmer = SnowballStemmer('english')
with open(sys.argv[1]) as f:
  for line in f:
    query, term = line.strip().split(',')
    print(','.join([query, stemmer.stem(term)]))
