'''
    spam.py builds a spam filter based on Paul Graham's A Plan for Spam.  CS 344 Calvin College

 @author:  austin gibson
 @version: March 6, 2019

'''


spam_corpus = [["I", "am", "spam", "spam", "I", "am"], ["I", "do", "not", "like", "that", "spamiam"]]
ham_corpus = [["do", "i", "like", "green", "eggs", "and", "ham"], ["i", "do"]]

test_mail = ["I", "like", "green", "eggs", "and", "spam", "its", "good"]
test_mail2 = ["do", "i", "like", "green", "eggs", "and", "ham", "i", "do"]

#create hash tables
good = {}
bad = {}

#loop through corpuses and create hash tables with number of occurences
for i in range(len(spam_corpus)):
    for j in range(len(spam_corpus[i])):
        if spam_corpus[i][j] in good.keys():
            good[spam_corpus[i][j]] = good[spam_corpus[i][j]] + 1
        else:
            good[spam_corpus[i][j]] = 1

for i in range(len(ham_corpus)):
    for j in range(len(ham_corpus[i])):
        if ham_corpus[i][j] in bad.keys():
            bad[ham_corpus[i][j]] = bad[ham_corpus[i][j]] + 1
        else:
            bad[ham_corpus[i][j]] = 1

#ngood and nbad are the number of nonspam & spam messages
ngood = 0
nbad = 0

for key in good:
    ngood += good[key]

for key in bad:
    nbad += bad[key]

#calculate the probability email containing it is a spam
def token_prob(word):
    try:
        g = 2 * good[word]
    except:
        g = 0
    try:
        b = bad[word]
    except:
        b = 0
    if g + b > 1:
        return max(0.01, min(.99, min(1.0, b/nbad) / (min(1.0, g/ngood) + min(1.0, b/nbad))))
    else:
        return 0

#create list of individual probabilities
probList = []
for word in test_mail:
    probList.append(token_prob(word))

#make list interesting.  take probability distance from a neutral .5
interestingList = []
for prob in probList:
    interestingList.append(abs(prob - .5))


def probability(listofProbs):
    prod = 1
    compliments = 1
    for word in listofProbs:
        prod = prod * word
        compliments = compliments * (1 - word)
    return prod / (prod + compliments)


print("hash of good tokens: ")
print(good)
print("\n hash of bad tokens: ")
print(bad)
print("\n ngood: " + str(ngood))
print("\n nbad: " + str(nbad))
print("\ntest spam mail: ")
print(test_mail)
print("\nprobabilities of those words: ")
print(probList)
print("\nprobs distance from .5 : ")
print(interestingList)
print("\nfinal probability mail is spam: " + str(probability(interestingList)))