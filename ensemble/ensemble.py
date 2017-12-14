a = open('2742bench7.txt', 'r').readline()
b = open('1915bench7.txt', 'r').readline()
c = open('2382bench1.txt', 'r').readline()
d = open('3920bench1.txt', 'r').readline()
e = open('4021bench1.txt', 'r').readline()
f = open('443bench7.txt', 'r').readline()
g = open('760bench7.txt', 'r').readline()
h = open('803bench7.txt', 'r').readline()



deci = int(a)+int(c)+int(d)+int(e)+int(f)+int(g)+int(h)+int(b)
gold = open('gold_labels.txt', 'r').readline()

deci = str(deci)

for i in range(len(deci)):
    if int(deci[i]) <= 4:
        st += '0'
    else:
        st += '1'

x = 0
for i in range(len(st)):
    if st[i] == gold[i]:
        x += 1

print x*1.00000/len(gold)