import math
import numpy as np


def grad_f(x):
    return x

def grad_desc(x, iter=1000, learning_rate=0.5):
    i = 0
    while i<iter:
        dx = -learning_rate*grad_f(x)
        if np.abs(dx) < 1e-3:
            print(round(x**2,4))
            break
        print(round(x**2,4))
        x += dx
        i+=1
    return x


grad_desc(5)
grad_desc(-5)