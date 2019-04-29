%lab12 Austin Gibson
%12.3
%
% Implements a prolog program for inferences used to justify the burning
% of the witch in Monty Python.

witch(X):-burn(X).
burn(X):-madeOfWood(X).
madeOfWood(X):-floats(X).
floats(X):- weighsSameAsDuck(X).

weighsSameAsDuck(woman).

% ?- witch(woman).
% true.
