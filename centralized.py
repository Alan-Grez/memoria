import numpy as np
from pyomo.environ import *
from numpy.linalg import inv
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.pyplot as plt
import proyecciones as pro
import time

def modelo(Sigma, N = 3, M = 5, parametros = [50, 1000, 10000, 1000] , show = 0):
    
    ###########################
    # Parámetros
    ###########################
    model = ConcreteModel("Centralized Problema")

    model.I  = range(N)
    model.XI = range(M)


    # Probabilidades:
    Sigma = list(Sigma[0])
    
    
    # Parámetros funciones:
    I    = parametros[0]
    MC   = parametros[1]
    VOLL = parametros[2]
    D    = parametros[3]
    
    ###########################
    # Variables
    ###########################
    model.x1 = Var(model.I,           within = NonNegativeReals)
    model.x2 = Var(model.I, model.XI, within = NonNegativeReals)
    model.x3 = Var(model.XI,          within = NonNegativeReals)
    
    ###########################
    # Restricciones
    ###########################
    
    if True:
        ###########################
        # Capacidad
        ###########################
        def capacity_r(model, i, xi):
            return Sigma[xi]*(model.x1[i] - model.x2[i,xi]) >= 0

        model.CAPACITY = Constraint(model.I, model.XI, rule=capacity_r) 

        ###########################
        # Equilibrio
        ###########################
        def equilibrium_r(model, xi):
            return Sigma[xi]*(- D[xi] + model.x3[xi] + sum(model.x2[i,xi] for i in model.I)) >= 0

        model.EQUILIBRIUM = Constraint(model.XI, rule = equilibrium_r)

        ###########################
        # Demanda
        ###########################
        def demand_r(model, xi):
            return Sigma[xi]*(D[xi] - model.x3[xi]) >= 0
    
        model.DEMANDA = Constraint(model.XI, rule = demand_r) 

    
    ###########################
    # Variables Duales
    ###########################
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    
    
    ###########################
    # Funciones Objetivos
    ###########################
    model.obj = Objective( expr = sum(I[i]*model.x1[i] for i in model.I)                                                      \
                                + sum(Sigma[xi]*0.005*sum(MC[i][xi]*(model.x2[i,xi]**2) for i in model.I) for xi in model.XI) \
                                + sum(Sigma[xi]*(-1)*VOLL*(D[xi]-model.x3[xi]) for xi in model.XI)                                 )

    
    ###########################
    # Print
    ###########################
    if show == 1:
        model.pprint()
    
    
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
    # Variables
    ###########################
    # Verifica si la solución fue exitosa
    if result.solver.status == SolverStatus.ok:
        # Extrae los valores de las variables de decisión y los almacena en listas
        
        ###########################
        # Variables Primales
        ###########################
        x1_values = [model.x1[i].value for i in model.I]
        x2_values = [[model.x2[i, xi].value for i in model.I] for xi in model.XI]
        x3_values = [model.x3[xi].value for xi in model.XI]
        
        
        ###########################
        # Variables Duales
        ###########################
        dual_value_equilibrio = [model.dual[model.EQUILIBRIUM[xi]] for xi in model.XI]  # Suponiendo que hay solo una restricción con ese nombre
        dual_value_capacity = [[model.dual[model.CAPACITY[i, xi]] for xi in model.XI] for i in model.I]


        if show == 1:
            # Ahora 'x1_values', 'x2_values', y 'x3_values' contienen las soluciones en forma de listas
            print("Soluciones:")
            print("x1:", x1_values)
            print("x2:", x2_values)
            print("x3:", x3_values)
            print("Valor dual de la restricción 'EQUILIBRIO':", dual_value_equilibrio)
            print("Valor dual de la restricción 'CAPACITY':  ", dual_value_capacity)

    else:
        print("El solver no encontró una solución óptima.")
    
    
    return np.array(x1_values), np.array(x2_values), np.array(x3_values)[np.newaxis], np.array(dual_value_equilibrio), np.array(dual_value_capacity)