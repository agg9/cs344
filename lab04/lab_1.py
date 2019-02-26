'''
This module implements a simple classroom example of probabilistic inference
over the full joint distribution specified by AIMA, Figure 13.3.
It is based on the code from AIMA probability.py.

@author: austin gibson
@version Feb 21, 2013
'''

"""
4.1

Value of P(Cavity|catch): . 529
    math: P(a ^ b) / P(b)
        (.108 +.072)  / (.108+.072+.016+.144)

P(Coin2 | Coin1 =heads):
    True: .05       False: .05

    -Yes this confirmed by belief that flipping coins is a 50/50 chance.
"""



from probability import JointProbDist, enumerate_joint_ask

# The Joint Probability Distribution Fig. 13.3 (from AIMA Python)
P = JointProbDist(['Coin1', 'Coin2'])
Heads, Tails = True, False
P[Heads, Heads] = 0.25; P[Tails, Tails] = 0.25
P[Heads, Tails] = 0.25; P[Tails, Heads] = 0.25


PC = enumerate_joint_ask('Coin2', {'Coin1': Heads}, P)
print(PC.show_approx())

