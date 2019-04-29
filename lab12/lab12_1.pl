% a. i.
    killer(Butch). %butch is a killer is a fact
    married(Mia, Marsellus). %they are married so using building association.
    dead(Zed). % Zed is dead is a fact
    kills(Marcellus, X):-givesMiaFootMassage(X).
    %Marcellus kill X, if X gives mia a foot massage.
    loves(Mia, X):-goodDancer(X).
    %Mia loves X, if a X is a good dancer.
    eats(Jules, X):-nutritious(X); tasty(X).
    %Jules eats X if it is nutritious OR tasty.  Multiple if's combined into one.

%  ii.
%  1. wizard(ron):  True
%   ron is a wizard is a fact
%  2. witch(ron): undefined
%   undefined because witch()doesn't exist
%  3. wizard(hermione): False
%   hermoine is not a wizard.  Nothing provided for that.
%  4. witch(hermione): undefined
%   again undefined . doesn't exist
%  5. wizard(harry): True
%   Harry is a wizard. Since he is a quidditch player, he has a broom.
%   Since he has a broom and has a wand, he is a wizard.
%  6. wizard(Y):
%   Y=ron Y=harry
%   Will get all who are wizards, which are harry and ron.
%  7. witch(Y): undefined
%  again, undefined.


% b. Yes
%  "if p then q" given p, q is inferred
%   for exmaple.  If the weather is bad, I am sad.
    weather(bad).
    sad(Austin):-weather(bad).
%  Sad(Austin) returns True, therefore implementing modus ponens.


% c. Horn clauses
%   A horn caluse is a subset of propoitional logic and can help provide
%   increased efficiency with algorithms, thus making a program run
%   faster.


% d. No, prolog does not support this distrinction.
%  for example when telling a fact
%  wizard(ron)
%  and asking
%  ?- wizard(ron)
%  The operations are the same.


