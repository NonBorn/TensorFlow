import pip

a = 8
print a
a = 'hello'
print a
a = "hel'lo"
print a
a = 'he"llo'
a = 'he\llo'
print a
# gia na dw ti typou einai i metablhth
b = '''
dfcgvhb
'''
print b
print(type(a))

# gia delete
del (a)

# listes...den orizei pinakes
l = [1, 2, 3, 4]
print l

# i lista exei opoiodhpote typo dedomenwn
l = [1, 'hello', 2, 3, 4]
print l

# for item in something.... to bhma einai panta 1
for item in range(1, 10):
    print item

stmt = True
while (stmt):
    stmt = False

for item in l: print item

d = {}
d = dict()
print(type(d))
dir(d)

# import mia bibliothiki... poy ektypwnei omorfa t apotelesma
import pprint
from pprint import pprint

pprint('rwar')

pprint([i for i in range(1: 100)])

b= False
a= True
if a :
    print True
elif b :
    print 'b'
else:
    print False


# #na parw olous tou monous k na tous balw se lista
# l=[ i for i in range (1,1000) if (i%2==1)]
# print(l)

lista = ['a',1]
l2 = [item for item in lista if(type(item) is str)]
print l2