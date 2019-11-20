import numpy as np
#This function solves the corresponding q1,q2 by exhaustive test
# e.g. xd=[1.2,1.2]
def bruteforce_search(xd):
    threshold = 0.003
    for q1 in np.arange(0,2*np.pi,0.005):
        for q2 in np.arange(0,2*np.pi,0.005):
            f1=np.cos(q1)+np.cos(q1+q2)
            f2=np.sin(q1)+np.sin(q1+q2)
            if abs(f1-xd[0])<threshold and abs(f2-xd[1])<threshold:
                print(q1,q2)
                print(f1,f2)
bruteforce_search([1.6,1.2])