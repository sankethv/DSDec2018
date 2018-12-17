#Strings
name = "Sreeni Jilla"
namesq = 'xyz'

print(type(name))
print(type(namesq))

#access string content
print(name[0])
print(name[2:6])

#modify string content
name[0] = 'A' #Being string is immutable object, you can not edit

name + ' Hyd'
name = name + 'xyz'
print(name)
name = name.upper()
name = "mr"
name = name.capitalize()
print(name)

name = name.replace('Mr','Miss')
print(name)

isinstance(name, str) #True
isinstance(name, int) #False