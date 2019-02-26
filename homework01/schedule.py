'''
    schedule.py solves the problem of schedule courses in homework1
    - based on Zebra()
    @author austin gibson
    @version Feb 23, 2019
'''

from csp import CSP, min_conflicts, backtracking_search, AC3, parse_neighbors
from search import depth_first_graph_search

def Schedule():

    variables = 'cs108 cs112 cs212 cs214 cs232 cs262'.split()
    courses = ["cs108", "cs112", "cs212", "cs214", "cs232", "cs262"]
    TimeSlot = 'mwf900 tth1030 mwf1130 tth130 mwf130'.split()
    Rooms = 'nh253 sb382'.split()
    Assignments = {
        "cs108": "adams",
        "cs112": "vanderlinden",
        "cs212": "bailey",
        "cs214": "schuurman",
        "cs232": "adams",
        "cs262": "bailey"
    }

    #variables = Courses + Faculty + TimeSlot + Rooms
    #loop through courses and make course-possibleVal triples
    domains = {}
    for var in variables:
        domains[var] = []
        faculty = Assignments[var]
        for time in TimeSlot:
            for room in Rooms:
                domains[var].append([faculty, time, room])
    """
    neighbors = parse_neighbors(cs108:cs112; cs108:cs212; cs108:cs214; cs108:cs232; cs108:cs262;
                        cs112:cs212; cs112:cs214; cs112:cs232; cs112:cs262;
                        cs212:cs214; cs212:cs232; cs212:cs262;
                        cs214:cs232; cs214:cs262; cs232:cs262; , variables)
    """

    neighbors = {}

    for i in range(len(courses)):
        neighborslist = []
        for j in range(i + 1, len(courses)):
            neighborslist.append(courses[j])
        neighbors[courses[i]] = neighborslist

    def schedule_constraint(A, a, B, b, recurse=0):
        #checks time/faculty constraint
        if A[0] == B[0] and A[1] == A[1]:
            return False
        #checks time/room
        if A[1] == B[1] and A[2] == A[2]:
            return False
        return True

    return CSP(variables, domains, neighbors, schedule_constraint)


def print_solution(result):
    variables = 'cs108 cs112 cs212 cs214 cs232 cs262'.split()
    for h in variables:
        print('Course', h)
        for (var, val) in result.items():
            if val == h: print('\t', var)


puzzle = Schedule()


#result = depth_first_graph_search(puzzle)
#result = AC3(puzzle)
#result = backtracking_search(puzzle)
result = min_conflicts(puzzle, max_steps=1000)

if puzzle.goal_test(puzzle.infer_assignment()):
    print("Solution:\n")
    print_solution(result)
else:
    print("failed...")
    print(puzzle.curr_domains)
    puzzle.display(puzzle.infer_assignment())

