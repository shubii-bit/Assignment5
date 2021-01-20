import numpy as np
import matplotlib.pyplot as plt
import math as math


def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

def circ_gen(O,r):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ

#Triangle sides


#Coordinates of A
p = 0
q = 6
print(p,q)
#Triangle vertices
A = np.array([-2,3.45])
B = np.array([2,3.45])
P = np.array([0,5])
Q = np.array([0,-5])
O = np.array([0,0])



#Generating all lines
x_AB = line_gen(A,B)
x_PQ = line_gen(P,Q)
x_circ = circ_gen(O,4)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_PQ[0,:],x_PQ[1,:],label='$AB$')

plt.plot(x_circ[0,:],x_circ[1,:])



plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(O[0],O[1],'o')
plt.text(O[0] * (1 - 0.2), O[1] * (1) , 'O')





plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.plot()
plt.show()
