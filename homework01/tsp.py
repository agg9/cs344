'''
    tsp.py solves the traveling sales person problem using a local search.
    for class cs344
@author austin gibson
@version feb 22, 2019

'''

from search import Problem, hill_climbing, simulated_annealing, \
    exp_schedule
from random import randrange


class TravelingSalesPerson(Problem):

    def __init__(self, initial, distances):
        self.initial = initial
        self.distances = distances

    # list of possible swaps.  Never swap first or last
    def actions(self, state):
        actions = []

        for i in range(1,6):
            j = randrange(1, 6)
            while j == i:
                j = randrange(1,6)
            actions.append([i,j])

        return actions

    # swap actions from state to new state
    def result(self, state, action):
        new_state = state[:]
        temp_state = new_state[action[0]]
        new_state[action[0]] = new_state[action[1]]
        new_state[action[1]] = temp_state
        return new_state

    # total distance of traveled cities gives value
    def value(self, state):
        totalDistance = 0
        for i in range(len(state)):
            if i + 1 < len(state):
                city1 = state[i]
                city2 = state[i + 1]
                totalDistance += self.distances[(city1, city2)]
        return -totalDistance


if __name__ == '__main__':

    cities = ["Kings Landing", "Riverrun", "Winterfell", "Braavos", "Volantis", "Oldtown", "Kings Landing"]

    distances = {
        ("Kings Landing", "Braavos"): 22,
        ("Kings Landing", "Winterfell"): 5,
        ("Kings Landing", "Riverrun"): 8,
        ("Kings Landing", "Volantis"): 13,
        ("Kings Landing", "Oldtown"): 20,
        ("Braavos", "Kings Landing"): 22,
        ("Winterfell", "Kings Landing"): 5,
        ("Riverrun", "Kings Landing"): 8,
        ("Volantis", "Kings Landing"): 13,
        ("Oldtown", "Kings Landing"): 20,
        ("Braavos", "Winterfell"): 13,
        ("Braavos", "Riverrun"): 10,
        ("Braavos", "Volantis"): 14,
        ("Braavos", "Oldtown"): 12,
        ("Winterfell", "Braavos"): 13,
        ("Riverrun", "Braavos"): 10,
        ("Volantis", "Braavos"): 14,
        ("Oldtown", "Braavos"): 12,
        ("Winterfell", "Riverrun"): 8,
        ("Winterfell", "Volantis"): 7,
        ("Winterfell", "Oldtown"): 18,
        ("Riverrun", "Winterfell"): 8,
        ("Volantis", "Winterfell"): 7,
        ("Oldtown", "Winterfell"): 18,
        ("Riverrun", "Volantis"): 13,
        ("Riverrun", "Oldtown"): 6,
        ("Volantis", "Riverrun"): 13,
        ("Oldtown", "Riverrun"): 6,
        ("Volantis", "Oldtown"): 15,
        ("Oldtown", "Volantis"): 15
    }

    p = TravelingSalesPerson(cities, distances)
    print('Initial                       ' + str(cities)
          + '\tvalue: ' + str(p.value(cities)))

    hill_solution = hill_climbing(p)
    print('Hill-climbing solution        ' + str(hill_solution)
          + '\tvalue: ' + str(p.value(hill_solution))
          )

    annealing_solution = simulated_annealing(
        p,
        exp_schedule(k=20, lam=0.005, limit=1000)
    )
    print('Simulated annealing solution  ' + str(annealing_solution)
          + '\tvalue: ' + str(p.value(annealing_solution))
          )
