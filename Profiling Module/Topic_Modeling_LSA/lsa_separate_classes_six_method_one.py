from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import csv
import nltk

# files = ['data_lsa/final_dataset_machine_learning.csv', 'data_lsa/final_dataset_bigdata.csv', 'data_lsa/final_dataset_img_processing.csv']
files = ['ieee_dataset/machine_learning_dataset.csv', 'ieee_dataset/big_data_dataset.csv', 'ieee_dataset/data_mining_dataset.csv'
    ,'ieee_dataset/computer_vision_dataset.csv', 'ieee_dataset/bioiformatic_dataset.csv', 'ieee_dataset/artificial_intel_dataset.csv']
concept_words = {}
count = 1

for f in files:
    corpus = []
    with open(f, encoding="utf-8") as input_file:
        readCSV = csv.reader(input_file, delimiter=',')
        for row in readCSV:
            corpus.append(row[1])

    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    X = vectorizer.fit_transform(corpus)
    # print(vectorizer.get_feature_names())
    # print(X[1])
    # print(X.shape)

    lsa = TruncatedSVD(n_components = 1, n_iter = 20)
    lsa.fit(X)

    # Visualizing the concepts
    terms = vectorizer.get_feature_names()
    for i, comp in enumerate(lsa.components_):
        componentTerms = zip(terms,comp)
        sortedTerms = sorted(componentTerms,key=lambda x:x[1],reverse=True)
        sortedTerms = sortedTerms[:200]
        concept_words["Concept "+str(count)] = sortedTerms
    count = count + 1
    # print(concept_words)
print(concept_words)

test = []

with open('users/test_user_final_dr_sagara.csv', encoding="utf-8") as input_file:
    readCSV = csv.reader(input_file, delimiter=',')
    for row in readCSV:
        test.append(row[1])

limit = test.__len__()
print(limit)
output = []
all_scores = []
# Sentence Concepts
for key in concept_words.keys():

    sentence_scores = []

    for sentence in test:
        words = nltk.word_tokenize(sentence)
        score = 0
        for word in words:
            for word_with_score in concept_words[key]:
                if word == word_with_score[0]:
                    score += word_with_score[1]
        sentence_scores.append(score)
        all_scores.append(score)
    print("\n"+key+":")

    sum = 0
    for sentence_score in sentence_scores:
        print(sentence_score)
        sum = sum + sentence_score
    avg = (sum / limit) * 2
    output.append(avg)

# print(output)
# print(sentence_scores)
# print(all_scores)
results_len = len(all_scores)
# print(results_len)

ml = []
bd = []
dm = []
cv = []
bio = []
ai = []

count = 1
for score in all_scores:
    if(count <= (results_len/6)*1):
        ml.append(score)
        count = count + 1
    elif(count <= (results_len/6)*2):
        bd.append(score)
        count = count + 1
    elif(count <= (results_len/6)*3):
        dm.append(score)
        count = count + 1
    elif (count <= (results_len / 6) * 4):
        cv.append(score)
        count = count + 1
    elif (count <= (results_len / 6) * 5):
        bio.append(score)
        count = count + 1
    elif (count <= (results_len / 6) * 6):
        ai.append(score)
        count = count + 1
    else:
        print('Process finished with error')

a = []
b = []
c = []
d = []
e = []
f = []

for i in range(0, int(results_len/6)):
    # if(ml[i] == 0 and bd[i] == 0 and img[i] == 0):
    #     break

    if(ml[i] > bd[i] and ml[i] > dm[i] and ml[i] > cv[i] and ml[i] > bio[i] and ml[i] > ai[i]):
        a.append(ml[i])

    elif(bd[i] > ml[i] and bd[i] > dm[i] and bd[i] > cv[i] and bd[i] > bio[i] and bd[i] > ai[i]):
        b.append(bd[i])

    elif (dm[i] > ml[i] and dm[i] > bd[i] and dm[i] > cv[i] and dm[i] > bio[i] and dm[i] > ai[i]):
        b.append(bd[i])

    elif(cv[i] > ml[i] and cv[i] > bd[i] and cv[i] > dm[i] and cv[i] > bio[i] and cv[i] > ai[i]):
        d.append(cv[i])

    elif (bio[i] > ml[i] and bio[i] > bd[i] and bio[i] > dm[i] and bio[i] > cv[i] and bio[i] > ai[i]):
        e.append(cv[i])

    elif (ai[i] > ml[i] and ai[i] > bd[i] and ai[i] > dm[i] and ai[i] > cv[i] and ai[i] > bio[i]):
        d.append(cv[i])

ml_len = (len(a))
bd_len = (len(b))
dm_len = (len(c))
cv_len = (len(d))
bio_len = (len(e))
ai_len = len(f)

print(ml_len)
print(bd_len)
print(dm_len)
print(cv_len)
print(bio_len)
print(ai_len)

ml_score = (ml_len / int(results_len/6)) * 100
bd_score = (bd_len / int(results_len/6)) * 100
dm_score = (dm_len / int(results_len/6)) * 100
cv_score = (cv_len / int(results_len/6)) * 100
bio_score = (bio_len / int(results_len/6)) * 100
ai_score = (ai_len / int(results_len/6)) * 100

print(str(ml_score) + ', ' + str(bd_score) + ', ' + str(dm_score) + ', ' + str(cv_score) + ', ' + str(bio_score) + ', ' + str(ai_score))
