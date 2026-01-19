#!/usr/bin/env python
# coding: utf-8

# In[1]:


# numpy import statement
import numpy as np
np.random.seed(1210)


# **Note**: Each question has some test cases, to help you figure out if your implementation is correct
# 

# ## Algorithm questions

# Implement the following algorithms according to their docstring.
# 
# You may **only use on basic list/dictionary operations**, **do not use any fancy libraries that solve this problem for you!**

# In[5]:


def extract_odd_values(input_list):
    """
    Return a new list which includes odd elements from input_list.
    Arguments:
        input_list: list, a list of elements
    Returns:
        new_list: list, only including odd values of the input list
    """
    new_list = []

    for value in input_list:
        if value % 2 == 1:
            new_list.append(value)

    return new_list


# In[6]:


# test cases: extract_odd_values
l1 = [1,3,2,4,5,6]
l1_invert = extract_odd_values(l1)
print(l1) # should be [1,3,2,4,5,6]
print(l1_invert) # should be [1,3,5]
l2 = []
l2_invert = []
print(l2) # should be []
print(l2_invert) # should be []


# In[9]:


def has_duplicate(input_list):
    seen = []

    for value in input_list:
        if value in seen:
            return True
        else:
            seen.append(value)

    return False


# In[10]:


# test cases for has_duplicate
l1 = ['A','T','C','G']
print(has_duplicate(l1)) # should be False
l2 = ['T','T','C','A']
print(has_duplicate(l2)) # should be True


# In[11]:


def invert_dict(input_dict):
    new_dict = {}

    for key in input_dict:
        value = input_dict[key]

 # If this value is already a key in the inverted dict, it's a duplicate
        if value in new_dict:
            raise ValueError("Duplicate value found in input_dict")

        new_dict[value] = key

    return new_dict


# In[12]:


# test cases for invert_dict
d1 = {"A":0,"C":1,"T":2,"G":3}
d1_invert = invert_dict(d1)
print("d1: ", d1) # should be {"A":0,"C":1,"T":2,"G":3}
print("d1_invert: ", d1_invert) # should be {0:"A", 1:"C", 2:"T", 3:"G"}
d2 = {"A":0,"C":0,"T":2,"G":3}
d2_invert = invert_dict(d2) # should raise ValueError
print("d2: ", d2) # should not be executed
print("d2_invert: ", d2_invert) # should not be executed


# In[14]:


def mutagenesis(seq, mut_dict):
    new_seq = ""

    for base in seq:
        # Step 1: validate DNA base
        if base not in ["A", "C", "T", "G"]:
            raise ValueError("Invalid DNA sequence")

        # Step 2: mutate if base exists in mut_dict
        if base in mut_dict:
            new_seq += mut_dict[base]
        else:
            new_seq += base

    return new_seq


# In[15]:


# test cases for mutatagenesis
# A becomes T, T becomes A, other bases are unchanged
mut_dict = {
    "A": "T",
    "T": "A",
}
seq1 = "AACTGACTTAGCA"
mut_seq1 = mutagenesis(seq1,mut_dict)
print(seq1) # should be "AACTGACTTAGCA"
print(mut_seq1) # should be "TTCAGTCAATGCT"
seq2 = "ACTGM5"
mut_seq2 = mutagenesis(seq2,mut_dict) # should raise ValueError
print(seq2) # should not be executed
print(mut_seq2) # should not be executed


# ## Numpy Programming Question

# ### Mins and Sums
# 
# In this question, you must implement two related functions. The first function, `min_then_sum`, computes the min over user-specified axes `axes` for two different arrays, then it adds the resulting arrays together (elementwise). The second function, `sum_then_min`, does the same thing but in opposite order: first it computes the sum over user-specified axes `axes`, then it takes the elementwise min of the resulting arrays.

# In[16]:


def min_then_sum(array_1, array_2, axes):
    """
    Take min of array_1 and array_2 over axes, then return the elementwise
    sum of the resulting arrays.

    Hint: if a, b are arrays with same shape, a + b is elementwise sum of a and b.
    """
    assert array_1.shape == array_2.shape

    min1 = np.min(array_1, axis=axes)
    min2 = np.min(array_2, axis=axes)

    return min1 + min2


# In[17]:


# min_then_sum tests
a1 = np.array([1.,2.,3.])
b1 = np.array([1.,1.,1.])
mts1 = min_then_sum(a1,b1,(0,))
print(mts1) # should be 2.0
a2 = np.arange(40).reshape(2,4,5) # 3-dimensional array with shape (2,4,5)
b2 = np.ones([40]).reshape(2,4,5) # 3-dimensional array with shape (2,4,5)
mts2 = min_then_sum(a2,b2,(0,2)) # 1-dimensional array with shape (4,)
print(mts2) # should be [ 1.  6. 11. 16.]


# In[18]:


def sum_then_min(array_1, array_2, axes):
    """
    Take sum of array_1 and array_2 over axes, then return the elementwise
    min of the resulting arrays.

    Hint: if a, b are arrays with same shape, np.minimum(a,b) is elementwise min
    of a and b.
    """
    assert array_1.shape == array_2.shape

    sum1 = np.sum(array_1, axis=axes)
    sum2 = np.sum(array_2, axis=axes)

    return np.minimum(sum1, sum2)


# In[19]:


# sum_then_min tests
a1 = np.array([1.,2.,3.])
b1 = np.array([1.,1.,1.])
stm1 = sum_then_min(a1,b1,(0,))
print(stm1) # should be 3.0
a2 = np.arange(40).reshape(2,4,5) # 3-dimensional array with shape (2,4,5)
b2 = np.ones([40]).reshape(2,4,5) # 3-dimensional array with shape (2,4,5)
stm2 = sum_then_min(a2,b2,(0,2)) # 1-dimensional array with shape (4,)
print(stm2) # should be [10. 10. 10. 10.]


# ### Computing Mean and Variance of a Discrete Distribution
# 
# A discrete distribution is defined over a list of values, often called its **support**. Each item in the support has an associated **probability mass function**, which we will shorten to **pmf**.
# 
# The mean of the distribution (a scalar) can be computed with the following equation:
# 
# $$ \texttt{mean} = \sum_i \texttt{pmf[i]} * \texttt{support[i]}$$
# 
# Similary, the variance of the distribution (a scalar) can be computed with this equation:
# 
# $$ \texttt{variance} = \sum_i \texttt{pmf[i]} * (\texttt{support[i]} - \texttt{mean})^2 $$
# 
# Implement these equations as the functions `compute_mean` and `compute_variance`.
# 
# **Note: you should not use a loop, instead use the `np.sum()` function!**

# In[20]:


def compute_mean(support, pmf):
    """
    Compute mean of the multinomial distribution

    Arguments:
        support: 1D numpy array, support of distribution
        pmf: 1D numpy array, probability mass function
    Returns:
        mean: float, the mean of the distribution
    """
    assert len(support) == len(pmf)

    return np.sum(pmf * support)


# In[21]:


# compute_mean tests
support1 = np.arange(5)
pmf1 = np.array([0.2,0.2,0.2,0.2,0.2])
mean1 = compute_mean(support1,pmf1)
print(mean1) # should be 2.0
support2 = np.arange(3)
pmf2 = np.array([0.3,0.2,0.5])
mean2 = compute_mean(support2,pmf2)
print(mean2) # should be 1.2


# In[22]:


def compute_variance(support,pmf):
    """
    Compute variance of the multinomial distribution
    Arguments:
        support: 1D numpy array, support of distribution
        pmf: 1D numpy array, probability mass function
    Returns:
        variance: float, the variance of the distribution
    """
    assert len(support) == len(pmf)

    mean = compute_mean(support,pmf)
    return np.sum(pmf * (support - mean) ** 2)


# In[23]:


# compute_variance tests
support1 = np.arange(5)
pmf1 = np.array([0.2,0.2,0.2,0.2,0.2])
var1 = compute_variance(support1,pmf1)
print(var1) # should be 2.0
support2 = np.arange(3)
pmf2 = np.array([0.3,0.2,0.5])
var2 = compute_variance(support2,pmf2)
print(var2) # should be 0.76


# Now, apply your `compute_mean` and `compute_variance` functions to estimate the mean and variance of a distribution from samples, by completing the `estimate_distribution` function below:

# In[24]:


def estimate_distribution(support, true_pmf, num_samples):
    """
    Draw IID samples from the distribution, estimate the distribution's parameters,
    and compute and print the true and estimated mean and variance.
    Arguments:
        support: 1D numpy array, support of distribution
        true_pmf: 1D numpy array, true probability mass function
        num_samples: int, number of samples to draw
    Returns:
        None
    """
    assert len(support) == len(true_pmf)
    assert num_samples > 0
    # compute true mean and variance using helper functions
    # YOUR CODE STARTS HERE
    true_mean = compute_mean(support, true_pmf)
    true_variance = compute_variance(support, true_pmf)
    # sample data from the true distribution
    # samples is an array of size [num_samples, len(support)]
    samples = np.random.multinomial(1, true_pmf, size=(num_samples,))
    # estimate pmf by taking mean across samples (the 0th axis)
    # YOUR CODE STARTS HERE
    est_pmf = np.mean(samples, axis=0)
    # compute estimated mean and variance using helper functions
    est_mean = compute_mean(support, est_pmf)
    est_variance = compute_variance(support, est_pmf)
    # print out the results
    print(f"True Statistics: Mean = {true_mean}, Variance = {true_variance}")
    print(f"Estimated Statistics: Mean = {est_mean}, Variance = {est_variance}")


# In[25]:


# estimate_distribution tests
# the estimated means and variances should be close to the real ones
# there might be some inaccuracy since the estimate has some randomness
num_samples = 1000
support1 = np.arange(5)
pmf1 = np.array([0.2,0.2,0.2,0.2,0.2])
estimate_distribution(support1,pmf1,num_samples)
support2 = np.arange(3)
pmf2 = np.array([0.3,0.2,0.5])
estimate_distribution(support2,pmf2,num_samples)


# In[ ]:




