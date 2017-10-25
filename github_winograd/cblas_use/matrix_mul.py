import numpy as np 

A = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
B = [[5,4],[3,2],[1,0]]

A = np.array(A)
B = np.array(B)
C = A.dot(B)
print C
