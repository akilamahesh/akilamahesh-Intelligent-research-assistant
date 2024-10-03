from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import csv
import nltk

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

    # visualizing the concepts
    terms = vectorizer.get_feature_names()
    for i, comp in enumerate(lsa.components_):
        componentTerms = zip(terms,comp)
        sortedTerms = sorted(componentTerms,key=lambda x:x[1],reverse=True)
        sortedTerms = sortedTerms[:200]
        concept_words["Concept "+str(count)] = sortedTerms
    count = count + 1
    # print(concept_words)
print(concept_words)

# test = {'kernel compar two text document kernel inner product featur space consist subsequ length subsequ order sequenc k charact occur text though necessarili contigu subsequ weight exponenti decay factor full length text henc emphasis occurr close contigu direct comput featur vector would involv prohibit amount comput even modest valu k sinc dimens featur space grow exponenti describ despit fact inner product effici evalu dynam program techniqu preliminari experiment comparison perform kernel compar standard word featur space kernel made show encourag result', 'increas research search engin cite literatur search hire decis accuraci system paramount import articl employ condit random field crf extract variou common field header citat research crf provid principl way incorpor variou local featur extern lexicon featur globl layout featur basic theori crf becom wellunderstood bestpractic appli realworld data requir addit explor make empir explor sever factor includ variat gaussian laplac hyperbolicilisubsub prior improv regular sever class featur base crf novel approach constraint corefer inform extract improv extract perform', 'approach analys deform left ventricl heart base parametr model give compact represent set point imag strategi track surfac sequenc cardiac imag follow track infer quantit paramet character left ventricl motion volum left ventricl eject fraction amplitud twist compon cardiac motion explain comput paramet model experiment result shown time sequenc two modal medic imag nuclear medicin xray comput tomographi video sequenc present result cdrom','huge amount data gener social medium emerg situat regard trove critic inform supervis machin learn techniqu earli stage disast challeng lack label data particular disast furthermor supervis model train label data prior disast may produc accur result address challeng domain adapt approach learn model predict target unlabel data target disast addit label data prior sourc disast use howev result model still affect varianc target domain sourc domain context propos hybrid featureinst adapt approach base matrix factor knearest neighbor algorithm respect propos hybrid adapt approach use select subset sourc disast data repres target disast select subset subsequ use learn accur supervis domain adapt bay classifi target disast word focu transform exist sourc data bring closer target data thu overcom domain varianc may prevent effect transfer inform sourc target combin select transform method use instanc featur respect show experiment propos approach effect transfer inform sourc target furthermor provid insight'}
# [ML, ML, IMG, BD]

test = []

with open('users/test_user_final_dr_upeksha.csv', encoding="utf-8") as input_file:
    readCSV = csv.reader(input_file, delimiter=',')
    for row in readCSV:
        test.append(row[1])

limit = test.__len__()
print(limit)
output = []
all_scores = []
# sentence Concepts
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
cutoff = 1.5

for i in range(0, int(results_len/6)):

    if(ml[i] > bd[i] and ml[i] > dm[i] and ml[i] > cv[i] and ml[i] > bio[i] and ml[i] > ai[i] and ml[i]> cutoff):
        a.append(ml[i])

    elif(bd[i] > ml[i] and bd[i] > dm[i] and bd[i] > cv[i] and bd[i] > bio[i] and bd[i] > ai[i] and bd[i] > cutoff):
        b.append(bd[i])

    elif (dm[i] > ml[i] and dm[i] > bd[i] and dm[i] > cv[i] and dm[i] > bio[i] and dm[i] > ai[i] and dm[i] > cutoff):
        b.append(bd[i])

    elif(cv[i] > ml[i] and cv[i] > bd[i] and cv[i] > dm[i] and cv[i] > bio[i] and cv[i] > ai[i] and cv[i] > cutoff):
        d.append(cv[i])

    elif (bio[i] > ml[i] and bio[i] > bd[i] and bio[i] > dm[i] and bio[i] > cv[i] and bio[i] > ai[i] and bio[i] > cutoff):
        e.append(cv[i])

    elif (ai[i] > ml[i] and ai[i] > bd[i] and ai[i] > dm[i] and ai[i] > cv[i] and ai[i] > bio[i] and bio[i] > cutoff):
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

# add all lengths to get common denominator
denominator = ml_len + bd_len + dm_len + cv_len + bio_len + ai_len

# calculate percentage for each category
ml_score = round(((ml_len / denominator) * 100), 2)
bd_score = round(((bd_len / denominator) * 100), 2)
dm_score = round(((dm_len / denominator) * 100), 2)
cv_score = round(((cv_len / denominator) * 100), 2)
bio_score = round(((bio_len / denominator) * 100), 2)
ai_score = round(((ai_len / denominator) * 100), 2)

print(str(ml_score) + ', ' + str(bd_score) + ', ' + str(dm_score) + ', ' + str(cv_score) + ', ' + str(bio_score) + ', ' + str(ai_score))
