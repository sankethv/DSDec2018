#Functions
#There is no begining and end in the python. Just indentation decides the begining and end.
def add(a,b,c):
    tmp = a+b
    tmp = tmp + c
    return tmp

print(add(10,20,30))

#Functions with optional arguments
def addWithOptionalArgs(a,b=10,c=20):
    tmp = a+b
    tmp = tmp + c
    return tmp

print(addWithOptionalArgs(5, 26))
print(addWithOptionalArgs(5,c=26))
print(addWithOptionalArgs(b=2,c=26)) #missing 1 required positional argument: 'a'

