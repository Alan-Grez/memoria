import numpy as np
from pyomo.environ import *


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
      
        




def P_D_pyo(x1_, x2_, P, gamma, show=0):
    # Esta función retorna la proyección de un punto arbitario de IR^NxIR^NM
    # al conjunto
    #
    #   D = {    x1 - x2_xi >= 0,  xi in range(M) &   x1, x2_xi  >= 0  }
    
    ###########################
    # Parametros del modelo
    ###########################
    N, M = x2_.shape

    model = ConcreteModel("Proyeccion onto positive capacity")
    
    x1_barra_P = x1_.T[0].tolist()
    x2_barra_P = x2_.tolist()
    proba      = P[0].tolist()

    model.I  = set(range(N))
    model.XI = set(range(M))

    ###########################
    # Variables
    ###########################
    model.x1 = Var(model.I,           within = NonNegativeReals)
    model.x2 = Var(model.I, model.XI, within = NonNegativeReals)


    ###########################
    # Restricciones
    ###########################
    def capacity_r(model, i, xi):
        #return proba[xi]*(model.x1[i] - model.x2[i,xi]) >= 0
        return model.x1[i] - model.x2[i,xi] >= 0
    
    model.CAPACITY = Constraint(model.I, model.XI, rule=capacity_r) 

    ###########################
    # Variables Duales
    ###########################
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    
    ###########################
    # Función objetivo
    ###########################
    model.obj = Objective( expr = (gamma**-1)*sum( 0.5*proba[xi]*sum((model.x2[i,xi] - x2_barra_P[i][xi])**2 for i in model.I) for xi in model.XI)\
                                + (gamma**-1)*0.5*sum((model.x1[i] - x1_barra_P[i])**2 for i in model.I) )    
    
    ###########################
    # Solver
    ###########################
    opt = SolverFactory('ipopt')
    result = opt.solve(model, tee=False) 

    ###########################
    # Display
    ###########################
    if show == 1:
        model.display()
        model.dual.display()

    ###########################
    # Resultados
    ###########################
    if result.solver.status == SolverStatus.ok:
        
        ###########################
        # Variables Primales
        ###########################
        x1_values = [model.x1[i].value for i in model.I]
        x2_values = [[model.x2[i, xi].value for xi in model.XI] for i in model.I]

        ###########################
        # Variables Duales
        ###########################
        dual_value_capacity = [[model.dual[model.CAPACITY[i, xi]] for xi in model.XI] for i in model.I]
            
    return np.array(x1_values)[:,np.newaxis], np.array(x2_values), np.array(dual_value_capacity)


def P_C_pyo(x2_, x3_, P, D, gamma, show=0):
    # Esta función retorna la proyección de un punto arbitario de IR^NxIR^NM
    # al conjunto
    #
    #   C = {  D_xi - x3_xi - 1I_{N}.T x2_xi >= 0,  xi in range(M) &   x3_xi, x2_xi  >= 0 & D_xi >= x3_xi }
    
    ###########################
    # Parametros del modelo
    ###########################
    N, M = x2_.shape

    model = ConcreteModel("Proyeccion onto positive equilibrium")
    
    x2_barra_P = x2_.tolist()
    x3_barra_P = x3_[0].tolist()
    proba      = P[0].tolist()
    D_         = D[0].tolist()

    model.I  = set(range(N))
    model.XI = set(range(M))

    ###########################
    # Variables
    ###########################
    model.x2 = Var(model.I, model.XI, within = NonNegativeReals)
    model.x3 = Var(model.XI,          within = NonNegativeReals)
    

    ###########################
    # Restricciones
    ###########################
    def equilibrium_r(model, xi):
        return proba[xi]*(- D_[xi] + model.x3[xi] + sum(model.x2[i,xi] for i in model.I)) >= 0
        #return - D_[xi] + model.x3[xi] + sum(model.x2[i,xi] for i in model.I) >= 0
    
    model.EQUILIBRIUM = Constraint(model.XI, rule = equilibrium_r)
    
    def demand_r(model, xi):
        return proba[xi]*( D_[xi] - model.x3[xi]) >= 0
        #return D_[xi] - model.x3[xi] >= 0
    
    model.DEMANDA = Constraint(model.XI, rule = demand_r) 

    ###########################
    # Variables Duales
    ###########################
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    
    ###########################
    # Función objetivo
    ###########################
    model.obj = Objective( expr = (gamma**-1)*sum( 0.5*proba[xi]*sum((model.x2[i,xi] - x2_barra_P[i][xi])**2 for i in model.I) for xi in model.XI)\
                                + (gamma**-1)*sum( 0.5*proba[xi]*(model.x3[xi] - x3_barra_P[xi])**2 for xi in model.XI) )    
    
    ###########################
    # Solver
    ###########################
    opt = SolverFactory('ipopt')
    result = opt.solve(model, tee=False) 

    ###########################
    # Display
    ###########################
    if show == 1:
        model.display()
        model.dual.display()

    ###########################
    # Resultados
    ###########################
    if result.solver.status == SolverStatus.ok:
        
            ###########################
            # Variables Primales
            ###########################
            x2_values = [[model.x2[i, xi].value for xi in model.XI] for i in model.I]
            x3_values = [ model.x3[xi].value    for xi in model.XI]
            
            ###########################
            # Variables Duales
            ###########################
            dual_value_equilibrio = [model.dual[model.EQUILIBRIUM[xi]] for xi in model.XI] 
            
    return np.array(x2_values), np.array(x3_values)[np.newaxis], np.array(dual_value_equilibrio)[np.newaxis]

def P_DcapC_pyo(x1_, x2_, x3_, P, D_, gamma, show=0):
    # Esta función retorna la proyección de un punto arbitario de IR^NxIR^NM
    # al conjunto
    #
    #   D = {    x1 - x2_xi >= 0,  xi in range(M) &   x1, x2_xi  >= 0  }
    
    ###########################
    # Parametros del modelo
    ###########################
    N, M = x2_.shape

    model = ConcreteModel("Proyeccion onto positive capacity inter equilibrium")
    
    x1_barra_P = x1_.tolist()
    x2_barra_P = x2_.tolist()
    x3_barra_P = x3_[0].tolist()
    D_         = D_[0].tolist()
    proba      = P[0].tolist()

    model.I  = set(range(N))
    model.XI = set(range(M))

    ###########################
    # Variables
    ###########################
    model.x1 = Var(model.I, model.XI, within = NonNegativeReals)
    model.x2 = Var(model.I, model.XI, within = NonNegativeReals)
    model.x3 = Var(model.XI,          within = NonNegativeReals)


    ###########################
    # Restricciones
    ###########################
    def capacity_r(model, i, xi):
        #return proba[xi]*(model.x1[i,xi] - model.x2[i,xi]) >= 0
        return model.x1[i,xi] - model.x2[i,xi] >= 0
    
    model.CAPACITY = Constraint(model.I, model.XI, rule=capacity_r) 
    
    def equilibrium_r(model, xi):
        #return proba[xi]*(- (D_[xi] - model.x3[xi]) + sum(model.x2[i,xi] for i in model.I)) >= 0
        return - (D_[xi] - model.x3[xi]) + sum(model.x2[i,xi] for i in model.I) >= 0 
    
    model.EQUILIBRIUM = Constraint(model.XI, rule=equilibrium_r)
    
    def demand_r(model, xi):
        #return proba[xi]*(D_[xi] - model.x3[xi]) >= 0
        return D_[xi] - model.x3[xi] >= 0 
    
    model.DEMANDA = Constraint(model.XI, rule = demand_r) 

    
    ###########################
    # Variables Duales
    ###########################
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    
    ###########################
    # Función objetivo
    ###########################
    model.obj = Objective( expr = (gamma**-1)*0.5*sum(proba[xi]*sum((model.x1[i,xi] - x1_barra_P[i][xi])**2 for i in model.I) for xi in model.XI) \
                                + (gamma**-1)*0.5*sum(proba[xi]*sum((model.x2[i,xi] - x2_barra_P[i][xi])**2 for i in model.I) for xi in model.XI) \
                                + (gamma**-1)*0.5*sum(proba[xi]*(model.x3[xi] - x3_barra_P[xi])**2                            for xi in model.XI) )    
    
    ###########################
    # Solver
    ###########################
    opt = SolverFactory('ipopt')
    result = opt.solve(model, tee=False) 

    ###########################
    # Display
    ###########################
    if show == 1:
        model.display()
        model.dual.display()

    ###########################
    # Resultados
    ###########################
    if result.solver.status == SolverStatus.ok:
        
        ###########################
        # Variables Primales
        ###########################
        x1_values = [[model.x1[i, xi].value for xi in model.XI] for i in model.I]
        x2_values = [[model.x2[i, xi].value for xi in model.XI] for i in model.I]
        x3_values = [ model.x3[xi].value    for xi in model.XI]

        ###########################
        # Variables Duales
        ###########################
        dual_value_capacity   = [[model.dual[model.CAPACITY[i, xi]] for xi in model.XI] for i in model.I]
        dual_value_equilibrio = [ model.dual[model.EQUILIBRIUM[xi]] for xi in model.XI]
        
    return np.array(x1_values), np.array(x2_values), np.array(x3_values)[np.newaxis],  np.array(dual_value_capacity), np.array(dual_value_equilibrio)[np.newaxis]  
        

def P_D(x1_barra: np.array, x2_barra: np.ndarray, P : np.array, gamma) -> tuple:
    """
        Input:
            - x1_barra: np.array(Nx1)
            - x2_barra: np.array(NxM)
            - P:        np.array(1xM)
        Output:
            - tuple (x1,x2) (np.array,np.array)
        Work:
            The function give the point x1, x2 that are the 
            projection of x1_barra, x2_barra over D without
            non-ancitipative policy and consider P as probability.
    """
    _, M = x2_barra.shape
    
    diff = np.maximum(x2_barra - x1_barra, 0)
  
    lambda_ = np.dot(diff, (np.identity(M) - 0.5*P).T)
   
    x1 = x1_barra + np.dot(lambda_, P.T)
    x2 = x2_barra - lambda_
    
    lambda_ = lambda_*(gamma**-1)
    
    return x1, x2, lambda_

def P_C_demanda(x2_barra: np.ndarray, x3_barra: np.ndarray, D: np.ndarray, gamma) -> tuple:
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
    N,_ = x2_barra.shape
    
    diff = np.maximum(D - x3_barra - x2_barra.sum(axis=0), 0)
    scale_factor = ((N + 1) ** -1)
    
    lambda_ = scale_factor * diff
    
    x2 = x2_barra + np.dot(np.ones((N,1)), lambda_)
    x3 = x3_barra + lambda_
    
    lambda_ = scale_factor * diff * (gamma**-1)
    
    return x2, x3, lambda_


def P_C(x2_barra: np.ndarray, x3_barra: np.ndarray) -> tuple:
    """
        Input:
            - x2_barra: np.array(NxM)
            - x3_barra: np.array(1xM)
            - D       : np.array(1xM) Término que simboliza la demanda
        Output:
            - tuple (x2,x3) (np.array,np.array)
        Work:
            The function give the point x2, x3 tmhat are the 
            projection of x2_barra, x3_barra over C.
    """
    N,_ = x2_barra.shape
    
    diff = np.maximum(x3_barra - x2_barra.sum(axis=0), 0)
    
    scale_factor = ((N + 1) ** -1)
    
    lambda_ = scale_factor * diff
    
    x2 = x2_barra + np.dot(np.ones((N,1)), lambda_)
    x3 = x3_barra - lambda_
    
    
    return x2, x3, lambda_


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
    
    lambda_ = 0.5*diff
    
    return x1, x2, lambda_



def P_N(x1_N_barra : np.array, P :np.array) -> np.array:
    """
        Input:
            - x1_N_barra: np.array(NxM)
        Output:
            - np.array x1
        Work:
            The function give the point x1 that is the 
            projection of x1_N_barra over N, the
            non-anticipative linear space.
    """
    _, M = x1_N_barra.shape
    return np.tile(np.matmul(x1_N_barra, P.T),(1,M))

def P_CcapD_NA(x1_N_barra : np.array, x2_barra : np.array, x3_barra : np.array, D: np.array, gamma) -> tuple:
    """
        Input:
            - x1_N_barra:      np.array(NxM)
            - x2_barra:        np.array(NxM)
            - x3_barra:        np.array(M)
            - P:               np.array(M)
        Output:
            - tuple np.array, np.array, np.array
        Work:
            The function give the point x1, x2, x3 that is the 
            projection of x1_N_barra, x2_barra, x3_barra over C
            intersection with D in combine with non-anticipative
            linear space.
    """
    N, M = x2_barra.shape
    
    diff_D = np.maximum(x2_barra - x1_N_barra, 0)
    diff_C = np.maximum(D - x3_barra - x2_barra.sum(axis=0)[np.newaxis,:] , 0)
    
    scale_factor = (2*(N + 2))**(-1)
    
    star_1 = scale_factor * ( np.dot((N+2)*np.identity(N) + np.ones((N,N)), diff_D) + 2*np.ones((N,M))*diff_C  )
    star_2 = scale_factor * ( np.dot(2*np.ones((1,N)), diff_D) + 4*diff_C  )
    
    x1 = x1_N_barra + star_1 +   0
    x2 = x2_barra   - star_1 + star_2
    x3 = x3_barra   +   0    + star_2
    
    return x1, x2, x3, star_1*(gamma**-1), star_2*(gamma**-1)

def P_CinterD_demanda(x1_N_barra : np.array, x2_barra : np.array, x2_barra_copy : np.array, x3_barra : np.array, D: np.ndarray, P) -> tuple:
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
                        
    return P_D_NA(x1_N_barra, x2_barra), P_C_demanda(x2_barra_copy, x3_barra, D, P)

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


def P_CcapD_NA_D(x1_barra : np.array, x2_barra : np.array, x3_barra : np.array, D : np.array, P : np.array) -> tuple:
    """
        Input:
            - x1_N_barra:      np.array(Nx1)
            - x2_barra:        np.array(NxM)
            - x3_barra:        np.array(1xM)
            - D:               np.array(1xM)
            - P:               np.array(1xM)
        Output:
            - tuple np.array, np.array, np.array
        Work:
            The function give the point x1, x2, x3 that is the 
            projection of x1_N_barra, x2_barra, x3_barra over C
            intersection with D in combine with non-anticipative
            linear space with D about demand
    """
    N,_ = x2_barra.shape
    
    scale_factor = (N + 2)**-1
    
    diff_D = np.maximum(x2_barra - x1_barra, 0)
    diff_C = np.maximum(D - x3_barra - x2_barra.sum(axis=0)[np.newaxis,:] , 0)
    
    star_1 = diff_D - 0.5*np.dot(diff_D, P.T) + np.dot( np.ones((N,N)), diff_D ) - (1-0.5*scale_factor)*np.dot( np.dot( np.ones((N,N)), diff_D ),  P.T)
    star_2 = diff_C - (N+1)*scale_factor*np.dot(diff_C, P.T)
    star_3 = diff_D.sum(axis = 0) - (N+1)*scale_factor*np.dot(diff_D.sum(axis = 0), P.T ) 
    star_4 = diff_C - N*scale_factor*np.dot(diff_C, P.T)
    
    lambda_ = star_1 + star_2
    mu_     = star_3 + star_4
    
    x1 = x1_barra + np.dot(lambda_, P.T) + 0
    x2 = x2_barra -        lambda_                   + mu_
    x3 = x3_barra +                  0               + mu_
    
    return x1, x2, x3, lambda_, mu_


