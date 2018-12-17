#Dictionaries are Key Value pairs
#Dictionary will be represented using {flower brackets} 

parameters =  { 'Height':10, 'Width':20}
type(parameters)

#access elements by key
print(parameters['Height'])
print(parameters['Width'])

#Alternate way to access elements by key
print(parameters.get('Height'))
print(parameters.get('Width'))

#Replace width value
parameters['Width'] = 50

#Access keys
print(parameters.keys())

#Access both key values pairs
print(parameters.items())