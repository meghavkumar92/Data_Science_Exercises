# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:20:44 2022

@author: Megha
"""

#1.	Create a string “Grow Gratitude”.

str1 = 'Grow Gratitude'
str1

#a)	How do you access the letter “G” of “Growth”?
print(str1[0])  #access using indexing

#b)	How do you find the length of the string?
print(len(str1)) #len() gives length of string

#c)	Count how many times “G” is in the string.
str_count = str1.count('G') #count() is used to count the occurence of the character 'G'
print(str_count)


#2. a)	Count the number of characters in the string.

str2 = 'Being aware of a single shortcoming within yourself is far more useful than being aware of a thousand in someone else.'

print(len(str2)) #len() gives the length of the string or count of characters in a string


#3.

str3 = 'Idealistic as it may sound, altruism should be the driving force in business, not just competition and a desire for wealth'

#a)	get one char of the word
print(str3[0]) ###################doubt

#b)	get the first three char
print(str3[:3])

#c)	get the last three char
print(str3[-3:])


#4. write a code to split on whitespace.

str4 = 'stay positive and optimistic'

print(str4.split())  #Splits the string at the specified separator, and returns a list. by default blankspace is the separator

#a)	The string starts with “H”
print(str4.startswith('H'))  #Returns true if the string starts with the specified value

#b)	The string ends with “d”
print(str4.endswith('d')) #Returns true if the string ends with the specified value

#c)	The string ends with “c”
print(str4.endswith('c')) #Returns true if the string ends with the specified value

#5.	Write a code to print 
i=0
for i in range(108):
    print('🪐')


#7. Create a string “Grow Gratitude” and write a code to replace “Grow” with “Growth of”

str7 = str1.replace('Grow', 'Growth of')    #Returns a string where a specified value is replaced with a specified value 
print(str7)  

    
#8. write a code to print the same story in a correct order.

str8 = '.elgnujehtotniffo deps mehtfohtoB .eerfnoilehttesotseporeht no dewangdnanar eh ,ylkciuQ .elbuortninoilehtdecitondnatsapdeklawesuomeht ,nooS .repmihwotdetratsdnatuotegotgnilggurts saw noilehT .eert a tsniagapumihdeityehT .mehthtiwnoilehtkootdnatserofehtotniemacsretnuhwef a ,yad enO .ogmihteldnaecnedifnocs’esuomeht ta dehgualnoilehT ”.emevasuoy fi yademosuoyotplehtaergfo eb lliw I ,uoyesimorp I“ .eerfmihtesotnoilehtdetseuqeryletarepsedesuomehtnehwesuomehttaeottuoba saw eH .yrgnaetiuqpuekow eh dna ,peels s’noilehtdebrutsidsihT .nufroftsujydobsihnwoddnapugninnurdetratsesuom a nehwelgnujehtnignipeelsecno saw noil A'

str_reversed = str8[::-1]  #string reverse built-in function is not available, so we use the pointer to move from -1 to position 0 in a string.
print(str_reversed)


