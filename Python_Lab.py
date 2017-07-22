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

# to null ths python
null = None
a = None
if ( a is None):
    print "None"

# pws grafw se arxeio
f = open('onomaaaaa.txt','w')
f.write('grammi 1\n')
f.write('grammi 2\n')
f.close()


# with as gia arxeia kai dinei nick name sto antikeimeno ...a=append
with open('onomeaaa.txt','w') as f :
    f.close()


# to diabazei olo to arxeio
with open('onomaaaaa.txt','r') as f :
    print f.read()
    f.close()

#sunarthseis
def func(a,b):
    return a+b
print func(1,3)

#sunarthseis
def func(a,b=5):
    return a+b

print func(1)
print func(1,6)
print func(a=1,b=6)
print func(b=1,6)



l=[1.1,2,3,4,5,6]
def fun (l):
    l=[ i for i in l if (i%2==1)]
    #
    l2 = []
    for i in l:
        if (i%2==1):
            l2.append(i)
    l = l2
    #
    f = open('./monoi.txt','w')
    f.write(str(l))
    f.close()


with open('./monoi.txt') as f :
    print f.read()
    f.close()


l=[1,2,3,4,5,6]
def fun (l):
    l=[ i for i in range (1,7) if (i%2==1)]
    with open('monoi.txt','w') as f :
        f.write(l)
        print f.read()
        f.close()

l=[1,2,3,4]
l=[ i for i in range (1,6) if (i%2==1)]
print l

l=[1,2,3,4]
l=[ i for i in range (1,5) if (i%2==1)]
print l

l = [1, 2, 3, 4]


def fun(l):
    l = [str(i) + '\n' for i in range(1, 5) if (i % 2 == 1)]
    with open('monoi.txt', 'w') as f:
        f.writelines(l)
        f.close()

    with open('monoi.txt', 'r') as f:
        print f.readlines()
        f.close()


fun (l)