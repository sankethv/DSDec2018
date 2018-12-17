#Entropy Calculations
import numpy as np

#For Dashed and solid lines 

#(-0.5714285714285714 * -0.8073549220576042) - (-0.42857142857142855 * -1.222392421336448) # -0.06253679653984678

# we have total 7 dashed triangles and squares out of that (4-triangles and 3-squares)
#I(dashed)
(-3/7*np.log2(3/7)) + (-4/7*np.log2(4/7)) # 0.9852281360342515

# we have total 7 solid triangles and squares out of that (1-triangles and 6-squares)
#I(solid)
(-6/7*np.log2(6/7)) + (-1/7*np.log2(1/7)) # 0.5916727785823275

I_res_Outline =  ((7/14*0.985) + (7/14*0.591)) #0.074
print(I_res_Outline)
Gain_outline = 0.940-0.788
print(Gain_outline)
############################################################################

#For Dots and yes or no
##total we have 6 dots 

#(-0.5714285714285714 * -0.8073549220576042) - (-0.42857142857142855 * -1.222392421336448) # -0.06253679653984678

#I( dotted) 3 triangles and 3 squares with dots
(-3/6*np.log2(3/6))+(-3/6*np.log2(3/6)) #1.0.

#I(non dotted)  2 triangles and 6 squares
(-2/8*np.log2(2/8))+(-6/8*np.log2(6/8)) # 0.811

I_res_Dot =  ((6/14*1.0) + (8/14*0.811)) #0.891
print(I_res_Dot)
Gain_Dot =0.940-0.891 ##0.048
print(Gain_Dot)
