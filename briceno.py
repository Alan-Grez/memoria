import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.pyplot as plt
import proyecciones as pro
import time

def Briceno_Arias_iteration(z1, z2, z2_copy, z3, gradient_of_g, P, gamma, lambda_n, D ):
    """
        Input:
            - z1     : np.array(NxM) decision variable
            - z2     : np.array(NxM) decision variable
            - z2_copy: np.array(NxM) decision variable
            - z3     : np.array(Mx1) decision variable
            - gradient_of_g: function has contain the gradien of the cost function
            - lambda_k: number between 0 and (4*beta - gamma)/(2*beta) where beta is 
                        the lipschitz constan of the gradien of h.
        Output:
            - tuple np.array, np.array, np.array, np.array
        Work:
            The function ompute one iteration over the Davis_Yin_algorithm
            with prior knowledge of gamma.
    """    
    
    # First step, xn = Pv zn
    x1, x2, x2_copy = pro.P_N1(z1, z2, z2_copy, P)
    x3 = z3
    
    # Unpackage of gradient
    g1, g2     , g3 = gradient_of_g(x1, x2,      x3, P)
    _ , g2_copy, _  = gradient_of_g(x1, x2_copy, x3, P)
    
    # Segundo step, yn = (xn-zn)/gamma
    y1      = (x1     - z1     )*(gamma**-1)
    y2      = (x2     - z2     )*(gamma**-1)
    y2_copy = (x2_copy- z2_copy)*(gamma**-1)
    y3      = (x3     - z3     )*(gamma**-1)
    
    # Third step, sn = xn - gamma * Pv ( gradient_of_g ) + gamma * yn
    (s1_aux, s2_aux, s2_copy_aux) = pro.P_N1(g1, g2, g2_copy, P)

    s1      = x1      - gamma*(s1_aux - y1)
    s2      = x2      - gamma*(s2_aux - y2)
    s2_copy = x2_copy - gamma*(s2_copy_aux - y2_copy)
    s3      = x3      - gamma*(g3 - y3)
    
    # Fourth step, pn = Pc\cap D sn
    (p1, p2), (p2_copy, p3) = pro.P_CinterD_demanda(s1, s2, s2_copy, s3, D)

    
    # Last step, zn+1 = zn + lambdan* (pn - xn)
    z1_k      = z1      + lambda_n * (p1 - x1)
    z2_k      = z2      + lambda_n * (p2 - x2)
    z2_copy_k = z2_copy + lambda_n * (p2_copy - x2_copy)
    z3_k      = z3      + lambda_n * (p3 - x3)
    
    return (z1_k, z2_k, z2_copy_k, z3_k), (x1, x2, x2_copy, x3)



def Briceno_Arias(N, M, number_iteration, frobenius_norm_of_MC, Grad_Phi, P, D, gamma=1e-3, lambdan=1e-3):

    z1      = np.random.random((N,M))
    z2      = np.random.random((N,M))
    z2_copy = np.random.random((N,M))
    z3      = np.random.random((1,M))
    
    
    beta = frobenius_norm_of_MC**(-1)
    #gamma = 2*beta*np.random.random(1)
    gamma = 2*beta*(0.65 + 0.35*np.random.random(1))
    alpha = np.maximum(2/3, (2*gamma)/(gamma+2*beta))
    
    x_list = [] 
    
    for k in range(number_iteration):
        
        #lambda_n = (1/alpha)*np.random.random(1)/(k**0.25+2)
        #lambda_n = (1/alpha)
        #lambda_n = 1
        #lambda_n = (1/alpha)*np.random.random(1)
        lambda_n = 0.95*(1/alpha)*np.random.random(1)+1
        #lambda_n =(1/alpha)*(0.75 + 0.5*np.random.random(1))
        
        (z1_k, z2_k, z2_copy_k, z3_k), (x1, x2, x2_copy, x3) = Briceno_Arias_iteration(z1, z2, z2_copy, z3, Grad_Phi, P, gamma, lambda_n, D)
        
        a = "D_factible"      if (x2 <= x1).all()                      else "D_infactible"
        b = "C_factible"      if (x2_copy.sum(axis=0) + x3 >= D).all() else "C_infactible"
        c = "Copy_factible"   if (x2_copy == x2).all()                 else "Copy_infactible"
        d = "NonAnt_factible" if (x1 == np.roll(x1, 1, axis=1)).all()  else "NonAnt_infactible"

        z1      = z1_k
        z2      = z2_k
        z2_copy = z2_copy_k
        z3      = z3_k        
        
        x_factible = "factible" if (a == "D_factible" and b == "C_factible" and d == "NonAnt_factible") else "infactible"

        
        x_list.append(((x1, x2, x2_copy, x3), x_factible))

        
    return x_list