m = 1291
l1 =[]
l2 =[]
for i in range(m):
  l1.append(i)
  l2.append(i*2)
import numpy as np
arr1 = np.array(l1)
arr2 = np.array(l2)
print(np.dot(arr1,arr2))