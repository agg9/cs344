"""
This module implements local search on a simple abs function variant.
The function is a linear function  with a single, discontinuous max value
(see the abs function variant in graphs.py).

@author: kvlinden
@version 6feb2013
"""
from tools.aima.search import Problem, hill_climbing, simulated_annealing, \
    exp_schedule, genetic_search
from random import randrange
import math
from timeit import default_timer as timer


class AbsVariant(Problem):
    """
    State: x value for the abs function variant f(x)
    Move: a new x value delta steps from the current x (in both directions)
    """

    def __init__(self, initial, maximum=30.0, delta=0.001):
        self.initial = initial
        self.maximum = maximum
        self.delta = delta

    def actions(self, state):
        return [state + self.delta, state - self.delta]

    def result(self, stateIgnored, x):
        return x

    def value(self, x):
        return math.fabs(x * math.sin(x))
        #return self.maximum / 2 - math.fabs(self.maximum / 2 - x)


if __name__ == '__main__':
    maximum = 30

    # Solve the problem using hill-climbing.
    bestHCcondition = 0
    hillClimbSum = 0
    for i in range(50):
        initial = randrange(0, maximum)
        p = AbsVariant(initial, maximum, delta=1.0)
        hill_solution = hill_climbing(p)
        hillClimbSum+=hill_solution

        if p.value(hill_solution) > p.value(bestHCcondition):
            bestHCcondition = hill_solution

    print('Hill-climbing best condition      x: ' + str(bestHCcondition)
              + '\tvalue: ' + str(p.value(bestHCcondition))
              )
    print('Hill-climbing average             x: ' + str(hillClimbSum / 50)
              + '\tvalue: ' + str(p.value(hillClimbSum /  50)))


    # Solve the problem using simulated annealing.

    bestSAcondition = 0
    simulatedAnnealingSum = 0
    for i in range(50):
        initial = randrange(0, maximum)
        p = AbsVariant(initial, maximum, delta=1.0)
        annealing_solution = simulated_annealing(
            p,
            exp_schedule(k=20, lam=0.005, limit=1000)
        )
        simulatedAnnealingSum+=annealing_solution

        if p.value(annealing_solution) > p.value(bestSAcondition):
            bestSAcondition = annealing_solution

    print('Simulated annealing best condition      x: ' + str(bestSAcondition)
              + '\tvalue: ' + str(p.value(bestSAcondition))
              )
    print('Simulated annealing average             x: ' + str(simulatedAnnealingSum / 50)
          + '\tvalue: ' + str(p.value(simulatedAnnealingSum / 50))
          )
