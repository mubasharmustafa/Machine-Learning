#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Solutions of Task 1


# In[3]:


# Task 1.1

nums = [3, 5, 7, 8, 12]

nums = [x**3 for x in nums]
nums


# In[7]:


# Task 1.2

anml_legs = dict()
animals, legs = ['parrot', 'goat', 'spider', 'crab'], [2, 4, 8, 10]

anml_legs = dict(zip(animals, legs))
anml_legs


# In[15]:


# Task 1.3

import sys
stdout = sys.stdout

anml_legs = {'parrot': 2, 'goat': 4, 'spider': 8, 'crab': 10}
count = 0

for _ in anml_legs.items():
    print(f'{_[0]} has {_[1]} legs')
    count += _[1]
print(f'Number of Total legs is: {count}')


# In[19]:


# Task 1.4

A = (3, 9, 4, [5, 6])
A[-1][0] = 8
print(A)


# In[23]:


# Task 1.5

A = (1, 2, 3, 4)
del A


# In[25]:


# Task 1.6

B = ('a', 'p', 'p', 'l', 'e')
print(f'"p" occurred {B.count("p")} times')


# In[27]:


# Task 1.7

B = ('a', 'p', 'p', 'l', 'e')
print(f'the index of "l" in B is: {B.index("l")}')


# In[ ]:


# Solutions of Task 2


# In[29]:


# Task 2.1

import numpy as np
A = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]

A_np = np.array(A)
A_np


# In[31]:


# Task 2.2
b = A_np[0:2, 0:2]
b


# In[35]:


# Task 2.3

C = np.zeros_like(A)
print(C)


# In[37]:


# Task 2.4

z = np.array([1, 2, 3])

for col in range(A_np.shape[1]):
    C[:, col] = A_np[:, col] + z
C


# In[39]:


X = np.array([[1,2],[3,4]])
Y = np.array([[5,6],[7,8]])
v = np.array([9,10])

# Task 2.5
X_add_Y = np.add(X, Y)
print(f'X + Y = {X_add_Y}\n')

# Task 2.6
X_mul_Y = np.multiply(X, Y)
print(f'X x Y = {X_mul_Y}\n')

# Task 2.7
sqrt_Y = np.sqrt(Y)
print(f'sq. root of Y = {sqrt_Y}\n')

# Task 2.8
dot_prod = np.dot(X, v)
print(f'Dot product of X and v = {dot_prod}\n')

# Task 2.9
x_sum_columns = np.sum(X, axis=0)
print(f'sum of columns of X = {x_sum_columns}\n')


# In[ ]:


# Solutions of Task 3


# In[43]:


# Task 3.1
def compute(distance, time):
    velocity = distance/time
    print(f"Velocity = {velocity:.3f}")
    return
compute(distance=120, time=76)


# In[49]:


# Task 3.2
import functools

even_num = [x for x in range(1,13) if x%2 == 0]
print(f'even numbers till 12: {even_num}')

def mul(lst):
    multiple = functools.reduce(lambda x, y: x*y, lst)
    print(f'list items multiple: {multiple}')
    return

mul(even_num)


# In[ ]:


# Solutions of Task 4


# In[53]:


import pandas as pd
df = pd.DataFrame([[1,6,7,7], [2,7,9,5], [3,5,8,2], [5,4,6,8], [5,8,5,8]], columns=['C1', 'C2', 'C3', 'C4'])
print(df)

# Task 4.1

print('\nPrinting only first 2 rows of df')
print(df[:2])


# In[69]:


# Task 4.2

print('\nPrinting first 2 columns of df')
print(df.iloc[:, 1:2])


# In[87]:


# Task 4.3

df.rename(columns={'C3': 'B3'}, inplace=True)
print('\nPrinting renamed column df')
print(df)


# In[89]:


# Task 4.4
print('\n Adding a new null column[sum] in df')
df['Sum'] = None
print(df)


# In[91]:


# Task 4.5

print('\nTaking Sum in sum column')
sums = df.sum(axis=1)
df['Sum'] = sums
print(df)


# In[109]:


# Task 4.6

# Read the dataframe file
dataframe = pd.read_csv("C:/Users/dell/hello_sample.csv")

# Task 4.7
print(dataframe)


# In[107]:


# Task 4.8

print(f'Printing last 2 rows of dataframe\n {dataframe.tail(2)}')


# In[111]:


# Task 4.9

print(f'\nPrinting dataframe info: \n')
dataframe.info()


# In[113]:


# Task 4.10

print(f'\nPrinting shape of dataframe: {dataframe.shape}')


# In[139]:


# Task 4.11

# Decending Order
sorted_dataframe_desc = dataframe.sort_values(by='Weight', ascending=False)
print(sorted_dataframe_desc)

# Ascending Orders
sorted_dataframe = dataframe.sort_values(by='Weight')
print(sorted_dataframe)


# In[125]:


# Task 4.12 (a)

null_check = dataframe.isnull()

print(null_check)


# In[133]:


# Task 4.12 (b)

cleaned_dataframe = dataframe.dropna()

print(cleaned_dataframe)

