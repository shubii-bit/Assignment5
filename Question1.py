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
a = 8

#Coordinates of A
p = 0
q = 6
print(p,q)
#Triangle vertices
A = np.array([p,q])
B = np.array([0,0])
C = np.array([a,0])
D = np.array([2.8,3.8 ])
E = np.array([4,0])
mid = np.array([2,3])
r = math.sqrt((2-4)**2+(3-0)**2)
T_1 = np.array([5.54,3.68])

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_BD = line_gen(B,D)
x_AE = line_gen(A,E)
x_circ = circ_gen(E,4)
x_circ2 = circ_gen(mid,r)
x_AT_1 = line_gen(A,T_1)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AE[0,:],x_AE[1,:])
plt.plot(x_BD[0,:],x_BD[1,:])
plt.plot(x_circ[0,:],x_circ[1,:])
plt.plot(x_circ2[0,:],x_circ2[1,:])
plt.plot(x_AT_1[0,:], x_AT_1[1,:])

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 + 0.03), C[1] * (1 - 0.1) , 'C')
plt.plot(D[0],D[1],'o')
plt.text(D[0] * (1 + 0.03), D[1] * (1 - 0.1) , 'D')
plt.plot(E[0] , E[1] , 'o')
plt.text(E[0] * (1 + 0.03), E[1] * (1 - 0.1) , 'O')
plt.plot(mid[0] , mid[1] , 'o')
plt.text(mid[0] * (1 + 0.03), mid[1] * (1 - 0.1) , 'E')
plt.plot(T_1[0],T_1[1],'o')
plt.text(T_1[0] * (1 + 0.03), T_1[1] * (1 - 0.1) , 'T_1')




plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.plot()
plt.show()
