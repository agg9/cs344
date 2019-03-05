'''
This module implements the Bayesian network shown in Exercise 5.2 for CS 344 at Calvin College

@author: austin gibson
@version march 2, 2019
'''

from probability import BayesNet, enumeration_ask, elimination_ask, gibbs_ask

# Utility variables
T, F = True, False

# From lab05 Exerecise 5.2
cancer = BayesNet([
    ('Cancer', '', 0.01),
    ('Test1', 'Cancer', {T: 0.9, F: 0.2}),
    ('Test2', 'Cancer', {T: 0.9, F: 0.2})
    ])

print("\n Cancer | positive results both test")
print(enumeration_ask('Cancer', dict(Test1=T,Test2=T), cancer).show_approx())
"""
False: .83   True: .17
I actually thought the result would be much higher with having 2 tests come back positive, but it was 
    only 17%.  This makes sense knowing that the chance of having cancer alone with very low at 1%.
    
    Work: P ( c | t1 , t2)
        =  alpha * P(t1|c)P(t2|c)p(c), P(t1|-c)P(t2|-c)P(-c)>
        =  alpha * <.9 * .9 * .01 ,  .2 * .2 * .99>
        =  alpha * <.0081, .0396>
        = .0081 / (.0081 + .0396 = .0477), .0396 / .0477
        = < .169, .83>
    
"""

print("\n Cancer | postive test 1 , negative test 2")
print(enumeration_ask('Cancer', dict(Test1=T,Test2=F), cancer).show_approx())
"""
False: 0.994    True:  0.00565
The results here also make sense.  If one of the tests comes back negative, it greatly decreased the probability
    of having cancer. 

    Work: P (c | t1 ^ -t2)
        = alpha * <P(c)P(t1|c)P(-t2|c), P(-c)P(t1|-c)P(-t2|-c)
        = alpha * <.01*.9*.1, .99*.2*.8>
        = alpha * <.0009, .1584>
        = .0009 / (.0009+.1584) , .1584 / .1593
        = <.005649, .994>
        
"""