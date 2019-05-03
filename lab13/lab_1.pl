% cs344 @ Calvin College.  lab13
% Austin Gibson


% 13.1
% a. Exercises.
% i. 3.2

directlyIn(katarina, olga).
directlyIn(olga, natasha).
directlyIn(natasha, irina).
in(X, Y):- directlyIn(X, Y).
in(X, Y):-
   directlyIn(X, Z),
   in(Z, Y).
% The base clause states that if X is directly in Y, then X is in Y.
% The recursion caluse states that if X is directly in Z and Z is in Y,
% then X is in Y. The system doesn't know whether Z is in Y until it
% reaches the recursive step of the system, and uses the facts provided
% by directlyIn to evaluate whether X is directly in Y.
%
% System is based on definitions from 3.1


% ACCIDENTLY DID 3.3.  Leaving it here since I built it and it works.
%directTrain(saarbruecken,dudweiler).
%directTrain(forbach,saarbruecken).
%directTrain(freyming,forbach).
%directTrain(stAvold,freyming).
%directTrain(fahlquemont,stAvold).
%directTrain(metz,fahlquemont).
%directTrain(nancy,metz).
%
% travelFromTo(X, Y):- directTrain(X, Y).
% travelFromTo(X, Y):-
%    directTrain(X, Z),
%    travelFromTo(Z, Y).
%


%ii. 4.5

tran(eins,one).
tran(zwei,two).
tran(drei,three).
tran(vier,four).
tran(fuenf,five).
tran(sechs,six).
tran(sieben,seven).
tran(acht,eight).
tran(neun,nine).

listtran([], []).
listtran([G|TailG], [E|TailE]) :-
   tran(G, E),
   listtran(TailG, TailE).
% ?- listtran([eins,neun,zwei],X).
%  returns: X = [one, nine, two].
% ?- listtran(X, [one,seven,six,two]).
%  returns: X = [eins,sieben, sechs, zwei].
%
%  Model is similar to the system build above, but takes into account
%  lists. By splitting the list into a Head and Tail, the system can
%  translate the head, pass the tail recursively and repeat by
%  continously splitting off the head and translating it using the tran
%  facts provided.
%


% b. Does Prolog implement a version of generalized modus ponens?
%  p -> q     :p   therefore q
q :- p.
p.
% ?- q.  true.
% Yes, prolog can implement a version of modus ponens. The output of the
% system results in q being true.

