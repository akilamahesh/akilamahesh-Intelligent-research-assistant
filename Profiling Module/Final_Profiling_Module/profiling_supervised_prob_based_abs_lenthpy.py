import csv
import scholarly
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# u_name = 'Thushari Silva'
# u_name = 'Supunmali Ahangama'
# u_name = 'Lochandaka Ranathunge'
# u_name = 'Upeksha Ganegoda'
u_name = 'Asoka Karunananda'
# u_name = 'Sagara Sumathipala'


def get_user_details(u_name):

    user_details = []
    last_id = ''
    with open('datasets/users.csv', encoding="utf-8") as users_data:
        users = csv.reader(users_data, delimiter=',')
        for row in users:
            last_id = row[0]

    u_id = int(last_id) + 1
    search_query = scholarly.search_author(u_name)
    author = next(search_query).fill()

    user_affliation = author.affiliation
    user_key_words = author.interests

    sentence = ''
    for word in user_key_words:
        sentence = sentence + word + ' '

    x = author.publications.__len__()

    if (x < 50):
        num_pub = x
    else:
        num_pub = 50

    abstracts = []
    for count in range(0, num_pub):
        pub = author.publications[count].fill()
        abst = pub.bib.get('abstract')
        abst = str(abst)

        abstracts.append(abst)

    user_details.append(u_id)
    user_details.append(u_name)
    user_details.append(user_affliation)

    print(user_details)
    preprocess_abstracts(abstracts)


def preprocess_abstracts(abstracts):

    bad_chars = ['<div class="gsc_vcd_value"', ' id="gsc_vcd_descr">', '<div class="gsh_small">', '.', '”', '“', 'ˆ', '<div class="gsh_csp">',
                 'gsh_cspExpert', 'gsh_csp', '</div>', '/', ':', '-', '(', ')', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ';', '[', ']', ',',
                 '"', '<', '>', '"', '=', '…', 'div', "’", '\'']

    bad_words = ['we', 'the', 'in', 'a', 'during', 'and', 'an', 'this', 'these', 'that', 'paper', 'review', 'more', 'our', 'purpose', 'from', 'to',
                 'ass', 'as', 'public', 'live', 'study', 'along', 'purpose', 'open', 'job', 'main', 'h', 'couple', 'year', 'service', 'user', 'new',
                 'present', 'report', 'use', 'within', 'exist', 'strong', 'using', 'work', 'task', 'always']

    preprocessed_abs = []
    for abstract in abstracts:
        for i in bad_chars:
            abstract = abstract.replace(i, '')

        tokens = nltk.word_tokenize(abstract)

        lower_words = [lw_word.lower() for lw_word in tokens]
        remove_bad_words = [word for word in lower_words if not word in bad_words]

        stop_words = set(stopwords.words('english'))
        with_out_stop_words = [w for w in remove_bad_words if not w in stop_words]

        # stem_abs = [stemmer.stem(word) for word in with_out_stop_words]

        final_abs = ''
        for word in with_out_stop_words:
            final_abs = final_abs + lemmatizer.lemmatize(word) + ' '
        preprocessed_abs.append(final_abs)

    predict_interests(preprocessed_abs)


def manual_inputs():

    abstracts = []

    with open('users/test_user_final_dr_sagara.csv', encoding="utf-8")as file:
        u_details = csv.reader(file, delimiter=',')

        for detail in u_details:
            abstracts.append(detail[1])

    preprocess_abstracts(abstracts)


def predict_interests(abstracts):

    model_file = 'model/ieee_SVC_kernal=rbf_g05_prob.sav'
    loaded_model = pickle.load(open(model_file, 'rb'))

    # with open('datasets/ieee_dataset_six_areas.csv', encoding="utf-8")as file:
    #     data = pd.read_csv(file)
    #
    # X_train, X_test, Y_train, Y_test = train_test_split(data.abstract, data.label, test_size=0.2)
    #
    # accuracy = loaded_model.score(X_test, Y_test)
    # print('Accuracy is ' + str(accuracy*100))
    #
    # print('confusion matrix :')
    # predictions = loaded_model.predict(X_test)
    # cm = confusion_matrix(Y_test, predictions)
    # print(cm)

    ml = []
    bd = []
    dm = []
    cv = []
    bioin = []
    ai = []

    all_prob = []

    for abstract in abstracts:
        # result = loaded_model.predict([abstract])
        prob = loaded_model.predict_proba([abstract])
        all_prob.append(prob)

    # print(all_prob)
    denominator = abstracts.__len__()
    print(denominator)
    limit = 0.49

    for p in all_prob:
        for i in range(0, 1):
            if(p[0][i] > limit):
                ml.append(p[0][i])
            if(p[0][i + 1] > limit):
                bd.append(p[0][i + 1])
            if (p[0][i + 2] > limit):
                dm.append(p[0][i + 2])
            if (p[0][i + 3] > limit):
                cv.append(p[0][i + 3])
            if (p[0][i + 4] > limit):
                bioin.append(p[0][i + 4])
            if (p[0][i + 5] > limit):
                ai.append(p[0][i + 5])

    ml_len = ml.__len__()
    bd_len = bd.__len__()
    dm_len = dm.__len__()
    cv_len = cv.__len__()
    bio_len = bioin.__len__()
    ai_len = ai.__len__()

    ml_score = round(((ml_len / denominator) * 100), 2)
    bd_score = round(((bd_len / denominator) * 100), 2)
    dm_score = round(((dm_len / denominator) * 100), 2)
    cv_score = round(((cv_len / denominator) * 100), 2)
    bio_score = round(((bio_len / denominator) * 100), 2)
    ai_score = round(((ai_len / denominator) * 100), 2)

    print(str(ml_score) + ', ' + str(bd_score) + ', ' + str(dm_score) + ', ' + str(cv_score) + ', ' + str(bio_score) + ', ' + str(ai_score))


# get_user_details(u_name)
manual_inputs()