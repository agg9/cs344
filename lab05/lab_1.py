'''
This module implements the Bayesian network shown in the text, Figure 14.2.
It's taken from the AIMA Python code.

@author: kvlinden, (modified by austin gibson)
@version march 2, 2019
'''

from probability import BayesNet, enumeration_ask, elimination_ask, gibbs_ask

# Utility variables
T, F = True, False

# From AIMA code (probability.py) - Fig. 14.2 - burglary example
burglary = BayesNet([
    ('Burglary', '', 0.001),
    ('Earthquake', '', 0.002),
    ('Alarm', 'Burglary Earthquake', {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001}),
    ('JohnCalls', 'Alarm', {T: 0.90, F: 0.05}),
    ('MaryCalls', 'Alarm', {T: 0.70, F: 0.01})
    ])



# Exercises 5.1
# < False, True >
#  i. P(Alarm | burglary ^ -earthquake)
print("\n Alarm | burglary ^ -earthquake")
print(enumeration_ask('Alarm', dict(Burglary=T, Earthquake=F), burglary).show_approx())
""" < .06, .94>.  The probability that an Alarm sounds given there was a burglary and no earthquake 
    is very likely, which makes sense because an alarm system is designed to catch a burglar"""

#  ii. P(John | burglary ^ -earthquake)
print("\n John | burglary ^ -earthquake")
print(enumeration_ask('JohnCalls', dict(Burglary=T, Earthquake=F), burglary).show_approx())
""" < .151, .849>.  The probability that John Calls, again given there was a burglary and no earthquake,
    is also high.  It isn't as high as the alarm, but John shouldn't be as reliable as an alarm system."""

#  iii. P(Burglary | alarm)
print("\n Burglary | alarm")
print(enumeration_ask('Burglary', dict(Alarm=T), burglary).show_approx())
""" < .626, .374>.   This problem computes the probabilty that there was a bulgarly given that the alarm
    had went off.  The answer makes sense because there could often be false alarms, or earthquakes"""

#  iv. P(Burglary | john ^ mary)
print("\n Burglary | john ^ mary ")
print(enumeration_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
""" < .716, .284>.    This problem computes the probability that there was a burglary given that
    both neighbors (John and Mary) called.  The answer again makes sense, because there could be false
    alarms, but also John and Mary and humans, not as reliable, but also independent of each other"""