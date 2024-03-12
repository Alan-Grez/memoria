import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.pyplot as plt
import proyecciones as pro
import time


def Davis_Yin_iteration(z1 : np.array, z2: np.array, z3: np.array, gradient_of_h, gamma, lambda_k, Demanda):
    """
        Input:
            - z1: np.array(1xN)   decision variable
            - z2: np.array(NxM)   decision variable
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
    
    xg_1, xg_2, lambda1 = pro.P_D(z1, z2)
    xg_3 = z3
    
    grad_x1, grad_x2, grad_x3 = gradient_of_h(xg_1, xg_2, xg_3)
    
    xf_1                =                 2*xg_1 - z1 - gamma*grad_x1
    xf_2, xf_3, lambda2 = pro.P_C_demanda(2*xg_2 - z2 - gamma*grad_x2,\
                                          2*xg_3 - z3 - gamma*grad_x3  , Demanda)
    
    z1 = z1 + lambda_k*(xf_1 - xg_1)
    z2 = z2 + lambda_k*(xf_2 - xg_2)
    z3 = z3 + lambda_k*(xf_3 - xg_3)
    
    return (xg_1, xg_2, xg_3), (xf_1, xf_2, xf_3), (z1, z2, z3), (lambda1, lambda2)



def Davis_Yin(N,M, frobenius_norm, cost_function, number_iteration, Demanda, sol_teo):
    
    # Punto inicial
    z1_k  = np.zeros((N, 1))
    z2_k  = np.zeros((N, M))
    z3_k  = np.zeros((1, M))
    
    # Hiper-parámetros
    
    beta = frobenius_norm**(-1)                     # Dependes of cost_function
    gamma = 2*beta*(0.75 + 0.5*np.random.random(1)) # Uniform distribution - U(2*beta*0.75, 2*beta)
    lambda_k = 1                                    # lambda_k in (0,(4*beta - gamma)/(2*beta = 1.75))
    
    print("Beta:", beta)
    print("Gamma:", gamma)
    print("Lambda_k:", lambda_k)
        
    xg_list  = []
    var_dual = []
    
    i=0
    Loss = 9e9

    while Loss > 0.8e-4:
    #for k in range(number_iteration):
              
        (xg1_k, xg2_k, xg3_k), (xf1_k, xf2_k, xf3_k), (z1, z2, z3), (lambda1, lambda2) = Davis_Yin_iteration(z1_k, z2_k, z3_k, cost_function, gamma, lambda_k, Demanda)
        
        Loss = LA.norm(xg1_k - sol_teo[0]) + LA.norm(xg2_k - sol_teo[1]) + LA.norm(xg3_k - sol_teo[2])
        Loss = Loss/(LA.norm(sol_teo[0])+ LA.norm(sol_teo[1]) + LA.norm(sol_teo[2]))
        
        z1_k = z1
        z2_k = z2
        z3_k = z3
        
        a           = "factible" if (xg1_k <= xg2_k).all()                       else "infactible"
        b           = "factible" if (xg2_k.sum(axis=0) + xg3_k >= Demanda).all() else "infactible"

        xg_factible = "factible" if all(cond == "factible" for cond in [a, b])   else "infactible"
        
        xg_list.append( ((xg1_k, xg2_k, xg3_k), xg_factible) )
        var_dual.append((lambda1 / gamma, lambda2 / gamma))
        
        i+=1
        print("Iteration:",i,"Loss:",Loss)
    return xg_list, var_dual, i