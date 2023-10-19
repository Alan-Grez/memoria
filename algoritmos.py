import numpy as np
from numpy.linalg import inv
import proyecciones as pro
import time

def help_algorithms():
    print("This '.py' document contains the iteration of algorithms")
    print("with and without non-anticipation, these are:")
    print()
    print("Davis-Yin algorithm")
    print("ADMM")
    print("Briceño-Arias")
    print("NewOne?")
    print()
  
    

def Davis_Yin_iteration(z1 : np.array, z2: np.array, z3: np.array, gradient_of_h, gamma, lambda_k):
    """
        Input:
            - z1: np.array(1xN)   decision variable
            - z2: np.array(NxM) decision variable
            - z3: np.array(Mx1)   decision variable
            - gradient_of_h: function has contain the gradien of the cost function
            - lambda_k: number between 0 and (4*beta - gamma)/(2*beta) where beta is 
                        the lipschitz constan of the gradien of h.
        Output:
            - tuple np.array, np.array, np.array
        Work:
            The function ompute one iteration over the Davis_Yin_algorithm
            with prior knowledge of gamma.
    """
    
    xg_1, xg_2 = pro.P_D(z1, z2) # xg_3 = z3
    
    grad_x1, grad_x2, grad_x3 = gradient_of_h#(xg_1, xg_2, z3)
    
    xf_2, xf_3 = pro.P_C(2*xg_2 - z2 - gamma*grad_x2, 2*z3   - z3 - gamma*grad_x3) # xf_1 = 2*xg_1 - z1 - gamma*grad_x1
    
    #lambda_k = some formula? 
    
    z_1 = z1 + lambda_k*(2*xg_1 - z1 - gamma*grad_x1 - xg_1)
    z_2 = z2 + lambda_k*(xf_2 - xg_2)
    z_3 = z3 + lambda_k*(xf_3 - z3  )
    
    return z_1, z_2, z_3


def ADMM_iteration(decision, state, dual, phi_z1, phi_z2, phi_z3, P, s=1e-3,t=1e-3,r=1e-3):
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
    x1, x2, x3                   = decision
    z1, z2                       = state       #Unpackage the variables
    lambda_1, lambda_2, lambda_3 = dual
    
    N, M = x1.shape
    
    scale_factor_1 = (s+t) ** -1
    scale_factor_2 = inv(np.einsum('i,jk -> ijk', r/P, np.ones((N,N))) - t*np.identity(N))
    scale_factor_3 = P/r

    diff_1 = -0.5*(x1 - x2) + z2 - lambda_3
    diff_2 =  0.5*(x2.sum(axis=0) - x3) - z1 + lambda_1
    
    # Start algorithm
    
    x1_k = scale_factor_1*( -np.tile(phi_1[:,None], (1,M)) + s*(pro.P_N(x1, P) - lambda_2) + t*(x1 + diff_1))
    x2_k = np.squeeze(scale_factor_2@((- phi_2 - (r/P)*np.tile(-x2.sum(axis=0) + diff_2,(N,1)) - t*(x2 - diff_1) ).T)[:,:,None]).T
    x3_k = scale_factor_3*(  phi_3 + (r/P)*(x3 + diff_2))
    
    z1_k = np.maximum( 0.5*(x2_k.sum(axis=0) - x3_k) + lambda_1 ,0)
    z2_k = np.maximum( 0.5*(x1_k - x2_k)   + lambda_3 ,0)
    
    lambda_1_k = lambda_1 + 0.5*(x2_k.sum(axis=0) - x3_k) - z1_k
    lambda_2_k = lambda_2 + x1_k - pro.P_N(x1_k, P)
    lambda_3_k = lambda_3 - 0.5*(x2 - x1) - z2_k
    
    return (x1_k, x2_k, x3_k), (z1_k, z2_k), (lambda_1_k, lambda_2_k, lambda_3_k)