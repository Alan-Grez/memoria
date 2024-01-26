import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.pyplot as plt
import proyecciones as pro
import time

def ADMM_iteration(decision, state, dual, phi_z1, phi_z2, phi_z3, P, D, e1, e2, e31, e32, r=1e-3):
    """
        Input:
            - decision = (x1,x2,x3): (np.array(NxM), np.array(NxM), np.array(Mx1) decision variable
            - state    = (z1, z2): (np.array(Mx1), np.array(NxM)) state variables
            - dual     = (lambda_1, lambda_2, lambda_3) (np.array(NxM), np.array(NxM), np.array(Mx1)) dual update variable
            
              Now will go the gradient of the objective functions, in this case we
              will assume there are linear, so their value is fix and a vector.
              
            - phi_z1: np.array(N) function has contain the gradient of the cost function over z1 variable 
            - phi_z2: np.array(NxM) function has contain the gradient of the cost function over z2 variable
            - phi_z3: np.array(M) function has contain the gradient of the cost function over zm3 variable
            
            - prob: np.array(M) np.array that contain the probability of each scenario
            
            - s,t,r: floats that are the augmented term of lagrangian. 1e-3 by default.
        Output:
            - tuple np.array, np.array, np.array
        Work:
            The function compute one iteration over the Davis_Yin_algorithm
            with prior knowledge of gamma.
    """
    Q1, B1 = phi_z1
    Q2, B2 = phi_z2
    Q3, B3 = phi_z3
    
    x1, x2, x3 = decision
    z1, z2, z3 = state    #Unpackage the variables
    y1, y2, y3 = dual
    
    N, M = x1.shape
    
  
    s1  = np.sign(z1-y1-(r**-1)*B1)
    s2  = np.sign(z2-y2-(r**-1)*B2)
    s31 = np.sign(z3-y3-(r**-1)*B3)
    s32 = np.sign((r**-1)*(Q3 + r)*D - (z3-y3-(r**-1)*B3))
    
    
    scale_factor_1 = r/(np.tile(np.diag(Q1),(M,1)).T + r + np.maximum(-s1*e1,0))
    scale_factor_2 = r/(np.diagonal(Q2, axis1=1, axis2=2).transpose() + r + np.maximum(-s2*e2,0))
    scale_factor_3 = r/(Q3+r)


    c1 = scale_factor_1 * (z1-y1-(r**-1)*B1)
    c2 = scale_factor_2 * (z2-y2-(r**-1)*B2)
    c3 = scale_factor_3 * (z3-y3-(r**-1)*B3)
    
    
    tk = (c2-c1+np.maximum(c1-c2,0))/((scale_factor_1+scale_factor_2)/r)
    
    
    # Start algorithm
    x1_k = c1 + (r**-1) * scale_factor_1 * tk
    x2_k = c2 - (r**-1) * scale_factor_2 * tk
    x3_k = ((Q3+r)*c3+np.maximum(-s32*e32,0)*D)/(Q3+r+np.maximum(-s31*e31,0)+np.maximum(-s32*e32,0))
    
    z1_k      = pro.P_N( y1 + x1_k, P)
    z2_k,z3_k = pro.P_C_demanda(y2 + x2_k, y3 + x3_k, D)
    
    y1_k = y1 + x1_k - z1_k
    y2_k = y2 + x2_k - z2_k
    y3_k = y3 + x3_k - z3_k
    
    return (x1_k, x2_k, x3_k), (z1_k, z2_k, z3_k), (y1_k, y2_k, y3_k)


def ADMM(Number_iteration, N, M, phi1, phi2, phi3, Sigma, D, e1, e2, e31, e32):
    
    (Q1,B1) = phi1
    (Q2,B2) = phi2
    (Q3,B3) = phi3
    
    x1     = np.zeros((N, M))
    x2     = np.zeros((N, M))
    x3     = np.zeros((1, M))
    
    z1     = np.zeros((N, 1))
    z2     = np.zeros((N, M))
    z3     = np.zeros((1, M))
    
    y1     = np.zeros((N, 1))
    y2     = np.zeros((N, M))
    y3     = np.zeros((1, M))

    r=0.5e2*0.90741
    
    ADMM_list = []
    
    for k in range(Number_iteration):
       
        (x1_k, x2_k, x3_k), (z1_k, z2_k,z3_k), (y1_k, y2_k, y3_k) = ADMM_iteration((x1, x2, x3), (z1, z2, z3), (y1, y2, y3), (Q1,B1), (Q2,B2), (Q3,B3), Sigma, D, e1, e2, e31, e32, r)
       
        x1 = x1_k
        x2 = x2_k
        x3 = x3_k
        
        z1 = z1_k
        z2 = z2_k
        z3 = z3_k
        
        y1 = y1_k
        y2 = y2_k
        y3 = y3_k
        
        
        a = "factible" if (x1 <= x2).all()                     else "infactible"
        b = "factible" if (x2.sum(axis=0) + x3 >= D).all()     else "infactible"
        d = "factible" if (x1 == np.roll(x1, 1, axis=1)).all() else "infactible"

        x = "factible" if all(cond == "factible" for cond in [a, b, d]) else "infactible"
            
        ADMM_list.append(((x1, x2, x3), (z1, z2, z3), (y1, y2, y3), x))
        
    return ADMM_list