
import numpy as np
import math
def real_cubic_solve(a, b, c, d):
        #adapted from https://github.com/shril/CubicEquationSolver/blob/master/CubicEquationSolver.py
        #returns only real root of cubic but will do so for matrix of coefficients. x3-x100 fasteer

        f = findF(a, b, c)                          # Helper Temporary Variable
        g = findG(a, b, c, d)                       # Helper Temporary Variable
        h = findH(g, f)                             # Helper Temporary Variable

        R = -(g / 2.0) + np.sqrt(h) # Helper Temporary Variable
        S =np.zeros(len(R))
        S[R>=0] = R[R>=0] ** (1 / 3.0)  
        S[R<0] = (-R[R<0])** (1 / 3.0) * -1

        T = -(g / 2.0) - np.sqrt(h)
        U =np.zeros(len(T))
        U[T>=0] = (T[T>=0]**(1 / 3.0))
        U[T<0] = ((-T[T<0])** (1 / 3.0)) * -1 

        x1 = (S + U) - (b / (3.0 * a))


        return x1           # Returning One Real Root only


    # Helper function to return float value of f.
def findF(a, b, c):
        return ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0


    # Helper function to return float value of g.
def findG(a, b, c, d):
        return (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a **2.0)) + (27.0 * d / a)) /27.0


    # Helper function to return float value of h.
def findH(g, f):
        return ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)

    

def solve(a, b, c, d):

    if (a == 0 and b == 0):                     # Case for handling Liner Equation
        return np.array([(-d * 1.0) / c])                 # Returning linear root as numpy array.

    elif (a == 0):                              # Case for handling Quadratic Equations

        D = c * c - 4.0 * b * d                       # Helper Temporary Variable
        if D >= 0:
            D = math.sqrt(D)
            x1 = (-c + D) / (2.0 * b)
            x2 = (-c - D) / (2.0 * b)
        else:
            D = math.sqrt(-D)
            x1 = (-c + D * 1j) / (2.0 * b)
            x2 = (-c - D * 1j) / (2.0 * b)
            
        return np.array([x1, x2])               # Returning Quadratic Roots as numpy array.

    f = findF(a, b, c)                          # Helper Temporary Variable
    g = findG(a, b, c, d)                       # Helper Temporary Variable
    h = findH(g, f)                             # Helper Temporary Variable

    if f == 0 and g == 0 and h == 0:            # All 3 Roots are Real and Equal
        if (d / a) >= 0:
            x = (d / (1.0 * a)) ** (1 / 3.0) * -1
        else:
            x = (-d / (1.0 * a)) ** (1 / 3.0)
        return np.array([x, x, x])              # Returning Equal Roots as numpy array.

    elif h <= 0:                                # All 3 roots are Real

        i = math.sqrt(((g ** 2.0) / 4.0) - h)   # Helper Temporary Variable
        j = i ** (1 / 3.0)                      # Helper Temporary Variable
        k = math.acos(-(g / (2 * i)))           # Helper Temporary Variable
        L = j * -1                              # Helper Temporary Variable
        M = math.cos(k / 3.0)                   # Helper Temporary Variable
        N = math.sqrt(3) * math.sin(k / 3.0)    # Helper Temporary Variable
        P = (b / (3.0 * a)) * -1                # Helper Temporary Variable

        x1 = 2 * j * math.cos(k / 3.0) - (b / (3.0 * a))
        x2 = L * (M + N) + P
        x3 = L * (M - N) + P

        return np.array([x1, x2, x3])           # Returning Real Roots as numpy array.

    elif h > 0:                                 # One Real Root and two Complex Roots
        R = -(g / 2.0) + math.sqrt(h)           # Helper Temporary Variable
        if R >= 0:
            S = R ** (1 / 3.0)                  # Helper Temporary Variable
        else:
            S = (-R) ** (1 / 3.0) * -1          # Helper Temporary Variable
        T = -(g / 2.0) - math.sqrt(h)
        if T >= 0:
            U = (T ** (1 / 3.0))                # Helper Temporary Variable
        else:
            U = ((-T) ** (1 / 3.0)) * -1        # Helper Temporary Variable

        x1 = (S + U) - (b / (3.0 * a))
        x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * math.sqrt(3) * 0.5j
        x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * math.sqrt(3) * 0.5j

        return np.array([x1, x2, x3])           # Returning One Real Root and two Complex Roots as numpy array.


