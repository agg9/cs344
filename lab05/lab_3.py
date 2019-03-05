'''
This module implements the Bayesian network shown in Exercise 5.3 for CS 344 at Calvin College

@author: austin gibson
@version March 2 , 2019
'''

from probability import BayesNet, enumeration_ask

# Utility variables
T, F = True, False

sunraise = BayesNet([
    ('Sunny', '', 0.7),
    ('Raise', '', 0.01),
    ('Happy', 'Sunny Raise', {(T, T): 1.0, (T, F): 0.7, (F, T): 0.9, (F, F): 0.1})
    ])

# Exercise 5.3 a
# P(Raise | sunny)
print("\n P(Raise | sunny)")
print(enumeration_ask('Raise', dict(Sunny=T), sunraise).show_approx())
"""
    False: .99    True: .01
    The answer makes sense because the these are two independent variables.  So the probability
        of a raise given its sunny is just the probability given a raise.
    Work: n/a
"""

# P(Raise | happy ^ sunny)
print("\n P(Raise | happy ^ sunny)")
print(enumeration_ask('Raise', dict(Happy=T, Sunny=T), sunraise).show_approx())
"""
    False: .986   True: .0142
    It makes sense that the answer is low because again sunny is independent from raise, and the chance of
        a raise is low. 
    Work: P( raise | happy ^ sunny)
        = alpha * < P(h|r s)P(r)P(s), P(h| -r s)P(-r)P(s) >
        = alpha * < 1 * .01 * .7 , .7 * .99 *.7>
        = alpha * <.07 , .4851>
        .007 + .4851 = .4921
        <.01422, .9857>
"""

# Exercise 5.3 b
# P(Raise | happy)
print("\n P(Raise | happy)")
print(enumeration_ask('Raise', dict(Happy=T), sunraise).show_approx())
"""
    False: .982   True:  .0185
    The probability of having a raise given your happy is low.  This makes sense because the probability
        of getting a raise is low overall.
"""

# P(Raise | happy ^ -sunny)
print("\n P(Raise | happy ^ -sunny)")
print(enumeration_ask('Raise', dict(Happy=T, Sunny=F), sunraise).show_approx())
"""
    False: .917   True:  .0833
    The probability of having a raise given your happy and it is not sunny is still low, but higher then when
        its given 'happy ^ sunny' and just 'happy'.  This makes sense because when it is not sunny, the probability
        of the agent being happy is very low unless he has a raise, which greatly increases his happiness.  But 
        because the probability of getting a raise is so low, the probability here is not higher. 
"""