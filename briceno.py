import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.pyplot as plt
import proyecciones as pro
import time

def norm_adjusted(x_sol, x_teo, sigma, M):
    ###########################
    # Cálculo de la norma del espacio
    ###########################
    x_sol_1, x_sol_2, x_sol_3 = x_sol
    x_teo_1, x_teo_2, x_teo_3 = x_teo
    sigma=sigma[0]
    
    return LA.norm(x_sol_1-x_teo_1) + sum([sigma[xi]*LA.norm(x_sol_2[:,xi]-x_teo_2[:,xi]) for xi in range(M)]) + sum([sigma[xi]*LA.norm(x_sol_3[:,xi]-x_teo_3[:,xi]) for xi in range(M)])


def norm_adjusted_N(x_sol, x_teo, sigma, M):
    x_sol_1, x_sol_2, x_sol_3 = x_sol
    x_teo_1, x_teo_2, x_teo_3 = x_teo
    sigma=sigma[0]
    
    return sum([sigma[xi]*LA.norm(x_sol_1[:,xi][:,np.newaxis]-x_teo_1) for xi in range(M)]) + sum([sigma[xi]*LA.norm(x_sol_2[:,xi]-x_teo_2[:,xi]) for xi in range(M)]) + sum([sigma[xi]*LA.norm(x_sol_3[:,xi]-x_teo_3[:,xi]) for xi in range(M)])


def Briceno_Arias_iteration(z1, z2, z3, gradient_of_g, P, gamma, lambda_n, D ):
    """
        Input:
            - z1     : np.array(NxM) decision variable
            - z2     : np.array(NxM) decision variable
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
    x1 = pro.P_N(z1, P)
    x2 = z2
    x3 = z3
    
    # Unpackage of gradient
    g1, g2, g3 = gradient_of_g(x1, x2, x3, P)
    
    # Segundo step, yn = (xn-zn)/gamma
    y1      = (x1 - z1)*(gamma**-1)
    y2      = (x2 - z2)*(gamma**-1)
    y3      = (x3 - z3)*(gamma**-1)
    
    # Third step, sn = xn - gamma * Pv ( gradient_of_g ) + gamma * yn
    s1      = x1 - gamma*(pro.P_N(g1, P) - y1)
    s2      = x2 - gamma*(g2             - y2)
    s3      = x3 - gamma*(g3             - y3)
    
    # Fourth step, pn = Pc\cap D sn
    p1, p2, p3, lambda1, lambda2 = pro.P_CcapD_NA(s1, s2, s3, D, gamma)

    
    # Last step, zn+1 = zn + lambdan* (pn - xn)
    z1_k      = z1 + lambda_n * (p1 - x1)
    z2_k      = z2 + lambda_n * (p2 - x2)
    z3_k      = z3 + lambda_n * (p3 - x3)
    
    return (z1_k, z2_k, z3_k), (x1, x2, x3), (lambda1, lambda2)



def Briceno_Arias_iteration_pyomo(z1, z2, z3, gradient_of_g, P, gamma, lambda_n, D ):
    """
        Input:
            - z1     : np.array(NxM) decision variable
            - z2     : np.array(NxM) decision variable
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
    x1 = pro.P_N(z1, P)
    x2 = z2
    x3 = z3
    
    # Unpackage of gradient
    g1, g2, g3 = gradient_of_g(x1, x2, x3, P)
    
    # Segundo step, yn = (xn-zn)/gamma
    y1      = (x1 - z1)*(gamma**-1)
    y2      = (x2 - z2)*(gamma**-1)
    y3      = (x3 - z3)*(gamma**-1)
    
    # Third step, sn = xn - gamma * Pv ( gradient_of_g ) + gamma * yn
    s1      = x1 - gamma*(pro.P_N(g1, P) - P*y1)
    s2      = x2 - gamma*(g2             - P*y2)
    s3      = x3 - gamma*(g3             - P*y3)
    
    # Fourth step, pn = Pc\cap D sn
    p1, p2, p3, lambda1, lambda2 = pro.P_DcapC_pyo(s1, s2, s3, P, D, gamma)

    
    # Last step, zn+1 = zn + lambdan* (pn - xn)
    z1_k      = z1 + lambda_n *(p1 - x1)
    z2_k      = z2 + lambda_n *(p2 - x2)
    z3_k      = z3 + lambda_n *(p3 - x3)
    
    return (z1_k, z2_k, z3_k), (x1, x2, x3), (lambda1/P, lambda2/P)




def Briceno_Arias(N, M, frobenius_norm_of_MC, Grad_Phi, P, D, sol_teo, gamma=1e-3, lambdan=1e-3):

    #z1      = np.random.random((N,M))
    #z2      = np.random.random((N,M))
    #z3      = np.random.random((1,M))
    
    z1      = np.zeros((N,M))
    z2      = np.zeros((N,M))
    z3      = np.zeros((1,M))
    
    beta = frobenius_norm_of_MC**(-1)
    #gamma = 2*beta*np.random.random(1)
    gamma = 2*beta*(0.85 + 0.15*np.random.random(1))
    
    print("beta: ",beta)
    print("gamma:",gamma[0])
    
    alpha = np.maximum(2/3, (2*gamma)/(gamma+2*beta))
    
    z_list = []
    x_list = []
    dual_l = []
    
    i=0
    Loss1 = 0.99
    Loss2 = 0.65
    
    #while Loss1 > 1.0e-10 and i < 1.0e4 and ((abs(Loss1) <= abs(Loss2)) or i < 1500):
    #while Loss1 > 1.0e-6 and i < 4.5e6:
    while i < 1.0e4:

        Loss2 = Loss1
        
        #lambda_n = 0.95*(1/alpha)*np.random.random(1)+1
        lambda_n =(1/alpha)*(0.85 + 0.15*np.random.random(1))
        
        #(z1_k, z2_k, z3_k), (x1, x2, x3), (lambda1, lambda2) = Briceno_Arias_iteration(z1, z2, z3, Grad_Phi, P, gamma, lambda_n, D)
        (z1_k, z2_k, z3_k), (x1, x2, x3), (lambda1, lambda2) = Briceno_Arias_iteration_pyomo(z1, z2, z3, Grad_Phi, P, gamma, lambda_n, D)
        
        
        Loss1 = norm_adjusted_N((x1,x2,x3), sol_teo, P, M)/norm_adjusted(sol_teo, (np.zeros((N,M)),np.zeros((N,M)),np.zeros((1,M))), P, M)
        #Loss1 = LA.norm(x1 - sol_teo[0]) + LA.norm(x2 - sol_teo[1]) + LA.norm(x3 - sol_teo[2])
        #Loss1 = Loss1/(LA.norm(sol_teo[0])+ LA.norm(sol_teo[1]) + LA.norm(sol_teo[2]))
        
        a  = "D_factible"      if (x2 <= x1).all()                      else "D_infactible"
        b  = "C_factible"      if (x2.sum(axis=0) + x3 >= D).all() else "C_infactible"
        d  = "NonAnt_factible" if (x1 == np.roll(x1, 1, axis=1)).all()  else "NonAnt_infactible"

        z1 = z1_k
        z2 = z2_k
        z3 = z3_k        
        
        x_factible = "factible" if (a == "D_factible" and b == "C_factible" and d == "NonAnt_factible") else "infactible"

        
        x_list.append(((x1, x2, x3), x_factible))
        dual_l.append((lambda1/P, lambda2/P))
        z_list.append((z1_k, z2_k, z3_k))
        
        i+=1 
        print("Iteration:",i,"lambda_n:",lambda_n[0],"Loss:",Loss1)
        
    print("beta: ",beta)
    print("gamma:",gamma[0])
    return x_list, z_list, dual_l, i