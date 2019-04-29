%Lab 12 Austin Gibson
%12.2
% a. i. exercises 2.1 (1,2,8,9,14)
% 1. bread = bread -> true.
% 2. 'Bread' = bread -> false.
% 8. food(x) = food(bread) -> X = bread
% 9. food(bread, X) = food(Y, sausage) -> X= Sausage, Y=bread
% 14. meal(food(bread, X) = meal(X, drink(beer)) -> false.
%
% a. ii.
% 1. ?- magic(house_elf).   : False
% 2. ?- wizard(harry).      : False
% 3. ?- magic(wizrd).       : False
% 4. ?- magic('McGonagall').: True
% 5. ?- magic(Hermoine).
%          Hermoine = dobby;
%          Hermoine = "McGonagall";
%          Hermoine = rita_skeeter.
%
%   It does this by searching the knowledge base for instatiations from
%   top to bottom, and repeating when it hits a match.
%

% b. Does infrence in propositional logic use unification? why or why
% not?
%
% Yes. Unification is the process of finding something that makes to
% given terms or values equal, and inference is done by applying
% unification to logical expressions in propositional logic.
%


% c. Yes.  Prolog does this by asserting all permises and the neagtion
% true.
