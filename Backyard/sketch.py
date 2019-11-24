'''
These are our options:
k-sparse
L1 regularization - Done - Didnt work that much
KL Regularization - Done - Worked
Custom optimization function
Check CBOW
Other todos:
test precision and recall: Done - they are correct but there is a theoretical bug in p@k since true lables might be
less than top k findings, therefore fome of predictions candidates will be wrong and cause error. Maybe we should set
a upper bound for k equal to number of true candidates
'''

