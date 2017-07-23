# -*- coding: utf-8 -*
my = {}
my['0'] = 7
my['1'] = 3
my['2'] = 5
import copy

#n = sorted(my, key=my.get, reverse=True)

m = copy.copy(my)

m[3]=77


i = 56
print i%10