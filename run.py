import os,sys
import numpy as np
from Bias_BIND import Bias_BIND

# parameters setting
shape=100
size=20
pattern=4
rbias=0.9
lbias=0.3
lerror=0.05

max_iter=20 # t_all
m1=50 # t_MF
m2=10 # t_BI


# simulate data
X=np.zeros((shape,pattern))
Y=np.zeros((shape,pattern))
for i in range(pattern):
	X[np.random.choice(shape,size,replace=False),i]=1
	Y[np.random.choice(shape,size,replace=False),i]=1

M_O=X.dot(Y.T)
b_x=np.random.uniform(low=lbias, high=rbias, size=shape)
b_y=np.random.uniform(low=lbias, high=rbias, size=shape)
M_x=np.zeros((shape,shape))
M_y=np.zeros((shape,shape))
for i in range(shape):
	M_x[i]=np.random.binomial(n=1, p=b_x[i], size=shape)
	M_y[i]=np.random.binomial(n=1, p=b_y[i], size=shape)

M_b=M_x*M_y.T
M_O=M_O+M_b
M_O=(M_O>0).astype(int)
M_e=np.random.binomial(1,lerror,size=shape*shape).reshape((shape,shape))
M_O=np.abs(M_O-M_e)

data={"O":M_O,"b_x":b_x,"b_y":b_y,"X":X,"Y":Y}


# run BABF

mask=np.random.binomial(1,0.9,shape*shape).reshape((shape,shape))
mask=mask.astype('bool')

comp=Bias_BIND(O=data["O"],K=pattern,mask=mask,max_iter=max_iter,m1=m1,m2=m2)
comp.run()

test=(comp.X.dot(comp.Y.T)>0).astype(int)
base=(data['X'].dot(data['Y'].T)>0).astype(int)
val=np.sum(np.abs(test-base))

print("reconstruct error is "+str(val))



