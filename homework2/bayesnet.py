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


# Compute P(Cloudy)
print("\n P(Cloudy")
print(enumeration_ask('Cloudy', dict(), cloudy).show_approx())

# Compute P(Sprinkler | cloudy)
print("\n P(Sprinkler | cloudy")
print(enumeration_ask('Sprinkler', dict(Cloudy=T), cloudy).show_approx())

# Compute P(Cloudy | sprinkler ^ not raining)
print("\n P(Cloudy | sprinkler ^ not raing")
print(enumeration_ask('Cloudy', dict(Sprinkler=T, Rain=F), cloudy).show_approx())

# Compute P(WetGrass | cloudy & sprinkler & rain)
print("\n P(WetGrass | cloudy & sprinkler & rain")
print(enumeration_ask('WetGrass', dict(Cloudy=T, Sprinkler=T, Rain=T), cloudy).show_approx())

# Compute P(Cloudy | grass not wet)
print("\n P(Cloudy | grass not wet")
print(enumeration_ask('Cloudy', dict(WetGrass=F), cloudy).show_approx())