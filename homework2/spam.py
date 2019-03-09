'''
    spam.py builds a spam filter based on Paul Graham's A Plan for Spam.  CS 344 Calvin College

 @author:  austin gibson
 @version: March 6, 2019

'''


spam_corpus = [["I", "am", "spam", "spam", "I", "am"], ["I", "do", "not", "like", "that", "spamiam"]]
ham_corpus = [["do", "I", "like", "green", "eggs", "and", "ham"], ["I", "do"]]

test_mail1 = ["I", "am", "spam", "but", "if", "I", "like", "green", "eggs", "and", "ham", "am", "I", "still", "spam"]
test_mail2 = ["I", "am", "spam", "but", "if", "I", "like", "green", "eggs", "and", "ham", "am", "I", "still", "spam",
                "no", "unless", "theres", "extra", "spam", "I", "am"]

#create hash tables
good = {}
bad = {}

#loop through corpuses and create hash tables with number of occurences
for i in range(len(spam_corpus)):
    for j in range(len(spam_corpus[i])):
        if spam_corpus[i][j] in bad.keys():
            bad[spam_corpus[i][j]] = bad[spam_corpus[i][j]] + 1
        else:
            bad[spam_corpus[i][j]] = 1

for i in range(len(ham_corpus)):
    for j in range(len(ham_corpus[i])):
        if ham_corpus[i][j] in good.keys():
            good[ham_corpus[i][j]] = good[ham_corpus[i][j]] + 1
        else:
            good[ham_corpus[i][j]] = 1

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

#create third hash to map token to probability
probhash = {}
for word in good:
    probhash[word] = token_prob(word)
for word in bad:
    probhash[word] = token_prob(word)


#getting new mail
def newMail(mailList):
    probList = []
    #get probability for tokens in new mail.  .4 if no probabilty
    if len(mailList) <= 15:
        for word in mailList:
            prob = token_prob(word)
            if prob != 0:
                probList.append(prob)
            else:
                probList.append(.4)
    # if mail tokens > 15.  Get 15 most interesting which is distance from .5
    else:
        unsortedList=[]
        for word in mailList:
            prob = token_prob(word)
            if prob != 0:
                unsortedList.append([prob, abs(prob - .5)])
            else:
                unsortedList.append([.4, abs(.4 - .5)])
        unsortedList.sort(key=lambda x: x[1], reverse=True)
        for i in range(15):
            probList.append(unsortedList[i][0])
    return probList

#determine if newMail is spam
def probability(listofProbs):
    prod = 1
    compliments = 1
    for prob in listofProbs:
        prod *= prob
        compliments *= 1 - prob
    return prod / (prod + compliments)

#print out hash tables, ngood, nbad
print("hash of good tokens: ")
print(good)
print("\n hash of bad tokens: ")
print(bad)
print("\n probability hash table: ")
print(probhash)
print("\n ngood: " + str(ngood))
print(" nbad: " + str(nbad))

#test new mail
newmail = newMail(test_mail1)
print("\n new mail: ")
print(test_mail1)
print("final probability mail is spam: " + str(probability(newmail)))

#test mail longer than len 15
newmail = newMail(test_mail2)
print("\n new mail: ")
print(test_mail2)
print("final probability mail is spam: " + str(probability(newmail)))