'''
This module implements the Bayesian network shown in figure 14.12.
For homework2, cs344 at Calvin College

@author: austin gibson
@version March 6, 2019
'''

from probability import BayesNet, enumeration_ask
#elimination_ask, gibbs_ask

# Utility variables
T, F = True, False

cloudy = BayesNet([
    ('Cloudy', '', 0.5),
    ('Sprinkler', 'Cloudy', {T: 0.1, F: 0.5}),
    ('Rain', 'Cloudy', {T: 0.8, F: 0.2}),
    ('WetGrass', 'Sprinkler Rain', {(T, T): 0.99, (T, F): 0.9, (F, T): 0.9, (F, F): 0.00}),
    ])


"""
2b. Independent values in full joint
     size = s^ N
     n = 4 variables
     2 ^ 4 = 16

2c. Independent values in bayesnet for domain
    Referencing the BayesNet Implement earlier, there are 9 pairs.
    
"""

# Compute P(Cloudy)
print("\n P(Cloudy")
print(enumeration_ask('Cloudy', dict(), cloudy).show_approx())
"""
Calculations:
    P(Cloudy) = <.5 , .5> taken from figure 14.12
"""
# Compute P(Sprinkler | cloudy)
print("\n P(Sprinkler | cloudy")
print(enumeration_ask('Sprinkler', dict(Cloudy=T), cloudy).show_approx())
"""
Calculations:
    P(Cloudy) = <.1, .9> again taken from figure 14.12
"""

# Compute P(Cloudy | sprinkler ^ not raining)
print("\n P(Cloudy | sprinkler ^ not raining")
print(enumeration_ask('Cloudy', dict(Sprinkler=T, Rain=F), cloudy).show_approx())
"""
Calculations:
    P(Cloudy | sprinkler ^ not raining)
        = alpha * < P(C)P(s|C)P(-r|C), P(-C)P(s|-C)P(-r|-C)
        = alpha * <.5 *.1*.2, .5*.5*.8>
        = alpha * <.01, .2>
        = .01 / (.01 + .2) , .2/ .21
        = < .0476, .952>
"""

# Compute P(WetGrass | cloudy & sprinkler & rain)
print("\n P(WetGrass | cloudy & sprinkler & rain")
print(enumeration_ask('WetGrass', dict(Cloudy=T, Sprinkler=T, Rain=T), cloudy).show_approx())
"""
Calculations:
    P(WetGrass | cloudy & sprinkler & r)
        = alpha * p(c) < P(s|c)P(r|c)*P(wg|sr), P(s|C)P(r|C)P(-wg|sr)>
        = alpha * .5 * <.1*.8*.99, .1*.8*.01>
        = alpha * .5 * < .0792 , .008>
        = <.0396 , .004>
        = .0396 / (.0396 + .0004) , .004 / .04
        = <.99 , .01>
"""
# Compute P(Cloudy | grass not wet)
print("\n P(Cloudy | grass not wet")
print(enumeration_ask('Cloudy', dict(WetGrass=F), cloudy).show_approx())
"""
Calculations:
    P(Cloudy | grass not wet)
        = alpha *P(C) <*P(sr|C)*P(-gw|sr) + P(-sr|C)P(-gw|-sr)+P(s-r|C)P(-gw|s-r)+P(-s-r|C)P(-gw|-s-r), same but not C>
        = alpha * .5 * <.08*.01 + .72*.1+ .02*.1+.18*1, .1*.01+.4*.1+.1*.1+.4*1>
        = alpha * .5 * <.2548 , .451>
        = <.1274 , .2255>
        = .1274 / (.1274+.2255) , .2255 / .3529
        = <.361 , .639>
"""