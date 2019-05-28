import xml.etree.ElementTree as ET

import sys


params = ET.parse(sys.argv[1]).getroot()
for query in params.findall('query'):
  num = query.find('number').text.strip()
  text = query.find('text').text.strip()
  for term in text.split():
    print(num, term, sep=',')
