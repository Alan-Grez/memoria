import numpy as np


def help_projections():
    print("This '.py' document contains the formulas on the sets C and D")
    print("with and without non-anticipation, these are:")
    print()
    print("D   = {x2_j <= x1, for all j} - without non-anticipation")
    print("C_j = {x2_j.sum() >= x3_j} ")
    print("D_j = {x2_j <= x1_j} - without non-anticipation")
    print("N   = {x1_j = x1_i, for all j,i} set of non-anticipation")
    print()
    print("So, in order, the projections are:")
    print("P_D, P_C, P_D_NA, P_N")
          

def P_D(x1_barra: np.array, x2_barra: np.ndarray) -> tuple:
    """
        Input:
            - x1_barra: np.array(Nx1)
            - x2_barra: np.array(NxM)
        Output:
            - tuple (x1,x2) (np.array,np.array)
        Work:
            The function give the point x1, x2 that are the 
            projection of x1_barra, x2_barra over D without
            non-ancitipative policy.
    """
    diff = np.maximum(x2_barra - x1_barra, 0) 
    scale_factor = 1/((x2_barra > x1_barra).sum(axis=1) + 1)
    
    x1 = x1_barra + (scale_factor * diff.sum(axis = 1))[:,np.newaxis]
    x2 = x2_barra + np.where(x2_barra <= x1_barra, 0, (scale_factor * diff.sum(axis=1))[:,np.newaxis] -  diff)
    
    lambda1 = diff - (scale_factor * diff.sum(axis=1))[:,np.newaxis]
    
    return x1, x2


def P_C_demanda(x2_barra: np.ndarray, x3_barra: np.ndarray, D: np.ndarray) -> tuple:
    """
        Input:
            - x2_barra: np.array(NxM)
            - x3_barra: np.array(1xM)
            - D       : np.array(1xM) Término que simboliza la demanda
        Output:
            - tuple (x2,x3) (np.array,np.array)
        Work:
            The function give the point x2, x3 that are the 
            projection of x2_barra, x3_barra over C.
    """
    N, M = x2_barra.shape
    
    diff = np.maximum(D-x3_barra - x2_barra.sum(axis=0),0)
    scale_factor = ((N + 1) ** -1) 
    
    x2 = x2_barra + scale_factor * diff
    x3 = x3_barra + scale_factor * diff
    
    lambda1 = scale_factor * diff
    
    return x2, x3

def P_C(x2_barra: np.ndarray, x3_barra: np.ndarray) -> tuple:
    """
        Input:
            - x2_barra: np.array(NxM)
            - x3_barra: np.array(1xM)
        Output:
            - tuple (x2,x3) (np.array,np.array)
        Work:
            The function give the point x2, x3 that are the 
            projection of x2_barra, x3_barra over C.
    """
    N,M = x2_barra.shape
    
    diff = np.maximum(x3_barra - x2_barra.sum(axis=0),0)
    scale_factor = ((N + 1) ** -1) 
    
    x2 = x2_barra + scale_factor * diff
    x3 = x3_barra - scale_factor * diff
    
    return x2, x3


def P_D_NA(x1_N_barra: np.array, x2_barra: np.array) -> tuple:
    """
        Input:
            - x1_barra: np.array(NxM)
            - x2_barra: np.array(NxM)
        Output:
            - tuple (x1,x2) (np.array,np.array)
        Work:
            The function give the point x1, x2 that are the 
            projection of x1_N_barra, x2_barra over D with
            non-anticipative.
    """
    diff = np.maximum(x2_barra - x1_N_barra, 0)
    
    x1 = x1_N_barra + 0.5 * diff
    x2 = x2_barra   - 0.5 * diff
    
    lambda1 = 0.5*diff
    
    return x1, x2



def P_N(x1_N_barra, proba):
    """
        Input:
            - x1_N_barra: np.array(NxM)
            - proba: np.array(M)
        Output:
            - np.array x1
        Work:
            The function give the point x1 that is the 
            projection of x1_N_barra over N, the
            non-anticipative linear space.
    """
    _, M = x1_N_barra.shape
    return np.tile(np.matmul(x1_N_barra, proba.T),(1,M))


def P_CinterD_demanda(x1_N_barra : np.array, x2_barra : np.array, x2_barra_copy : np.array, x3_barra : np.array, D: np.ndarray) -> tuple:
    """
        Input:
            - x1_N_barra:      np.array(MxN)
            - x2_barra:        np.array(MxN)
            - x2_barra_copy:   np.array(MxN)
            - x3_barra:        np.array(M)
        Output:
            - tuple np.array, np.array, np.array
        Work:
            The function give the point x1, x2, x2^bar, x3 that is the 
            projection of x1_N_barra, x2_barra, x2_barra_copy, x3_barra over C
            intersection with D in combine with non-anticipative
            linear space.
    """
                        
    return P_D_NA(x1_N_barra, x2_barra), P_C_demanda(x2_barra_copy, x3_barra, D)

def P_N1(x1_N_barra, x2_barra, x2_barra_copy, proba):
    """
        Input:
            - x1_N_barra:      np.array(NxM)
            - x2_barra:        np.array(MxN)
            - x2_barra_copy:   np.array(MxN)
            - proba:           np.array(M)
        Output:
            - np.array x1
            - np.array x2
            - np.array x2^bar
        Work:
            The function give the point x1, x2, x2^bar that is the 
            projection of x1_N_barra, x2_barra, x2_barra_copy over N1, the
            non-anticipative linear space combined with equality constraint of 
            x2 and x2^bar.
    """
    return P_N(x1_N_barra, proba), 0.5*(x2_barra+x2_barra_copy), 0.5*(x2_barra_copy+x2_barra)