# from lxml import etree
from bs4 import BeautifulSoup
from pathlib import Path

xmile_path = Path('TestModels\MTW pathology v0.3.stmx')
with open(xmile_path) as f:
    xmile_content = f.read().encode()
    # print(xmile_content)
    f.close()

# root = etree.fromstring(xmile_content)
# print(root.tag)

root = BeautifulSoup(xmile_content, 'xml')
stocks = root.findAll('stock')
for s in stocks:
    print(list(s.children))
    print(s.get('name'))
    # print(s.getattribute('name'))
    # print(dir(s))
