% cs344 @ Calvin College.  lab13
% Austin Gibson


% 13.1
% a. Exercises.
% i. 3.2

directTrain(saarbruecken,dudweiler).
directTrain(forbach,saarbruecken).
directTrain(freyming,forbach).
directTrain(stAvold,freyming).
directTrain(fahlquemont,stAvold).
directTrain(metz,fahlquemont).
directTrain(nancy,metz).

travelFromTo(X, Y):- directTrain(X, Y).
travelFromTo(X, Y):-
   directTrain(X, Z),
   travelFromTo(Z, Y).

% america explain.

% ii. 4.5

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
listtran([germanHead | germanTail], [englishHead | englishTail]):-
    tran(germanHead, englishHead),
    listtran(germanTail, englishTail).

