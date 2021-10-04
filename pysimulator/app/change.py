# -*- coding: utf-8 -*-
"""
@author: Salvatore
"""

with open('app.py') as f:
    lines = f.readlines()
    
lines2 = list()
for line in lines:
    lines2.append(line[6:])


with open('app.py', 'w') as f:
    for item in lines2:
        f.write("%s" % item)

