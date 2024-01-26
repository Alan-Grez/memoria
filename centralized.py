import numpy as np
from pyomo.environ import *
from numpy.linalg import inv
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.pyplot as plt
import proyecciones as pro
import time

def modelo(N = 3, M = 5, parametros = [50, 1000, 10000, 1000] , azar = 0, show = 0):

    model = ConcreteModel("Centralized Problema")

    model.I  = range(N)
    model.XI = range(M)


    # Probabilidades:
    if azar:
        Sigma = np.random.random((1,M))
        Sigma /= Sigma.sum()
    else:
        Sigma = np.ones((1,M)) 
        Sigma /= Sigma.shape[1] 

    Sigma = list(Sigma[0])

    # Parámetros funciones:
    I    = [ parametros[0] for i  in range(N)]
    MC   = [[parametros[1] for xi in range(M)] for i in range(N)]

    VOLL =   parametros[2]
    D    = [ parametros[3] for xi in range(M)]

    model.x1 = Var(model.I,           within = NonNegativeReals)
    model.x2 = Var(model.I, model.XI, within = NonNegativeReals)
    model.x3 = Var(model.XI,          within = NonNegativeReals)

    model.CAPACIDAD_PRODUCCION = ConstraintList()

    for xi in model.XI:
        for i in model.I:
            model.CAPACIDAD_PRODUCCION.add(0 <= model.x1[i] - model.x2[i,xi] )

    model.EQUILIBRIO = ConstraintList()

    for xi in model.XI:
        model.EQUILIBRIO.add( 0 <= sum(model.x2[i,xi] for i in model.I) - (D[xi] - model.x3[xi]) ) 

    model.DEMANDA = ConstraintList()

    for xi in model.XI:
        model.DEMANDA.add( model.x3[xi] <= D[xi] )


    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    model.obj = Objective( expr = sum(I[i]*model.x1[i] for i in model.I)                                                      \
                                + sum(Sigma[xi]*0.005*sum(MC[i][xi]*(model.x2[i,xi]**2) for i in model.I) for xi in model.XI) \
                                + sum(Sigma[xi]*(-1)*VOLL*(D[xi]-model.x3[xi]) for xi in model.XI)                                 )


    if show == 1:
        model.pprint()

    opt = SolverFactory('ipopt')
    result = opt.solve(model, tee=False) 

    if show == 1:
        model.display()
        model.dual.display()

    # Verifica si la solución fue exitosa
    if result.solver.status == SolverStatus.ok:
        # Extrae los valores de las variables de decisión y los almacena en listas
        x1_values = [model.x1[i].value for i in model.I]
        x2_values = [[model.x2[i, xi].value for i in model.I] for xi in model.XI]
        x3_values = [model.x3[xi].value for xi in model.XI]

        dual_value_equilibrio = [model.dual[model.EQUILIBRIO[xi+1]] for xi in model.XI]  # Suponiendo que hay solo una restricción con ese nombre


        if show == 1:
            # Ahora 'x1_values', 'x2_values', y 'x3_values' contienen las soluciones en forma de listas
            print("Soluciones:")
            print("x1:", x1_values)
            print("x2:", x2_values)
            print("x3:", x3_values)
            print("Valor dual de la restricción 'EQUILIBRIO':", dual_value_equilibrio)

    else:
        print("El solver no encontró una solución óptima.")
    
    
    return x1_values, x2_values, x3_values, dual_value_equilibrio