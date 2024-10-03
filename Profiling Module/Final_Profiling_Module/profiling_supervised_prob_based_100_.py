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
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter.ttk import Progressbar
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# u_name = 'Thushari Silva'
# u_name = 'Supunmali Ahangama'
# u_name = 'Lochandaka Ranathunge'
# u_name = 'Upeksha Ganegoda'
# u_name = 'Asoka Karunananda'
# u_name = 'Sagara Sumathipala'
# u_name = 'Thanuja Sandanayake'
# u_name = 'G. T. Weerasuriya'


def get_user_details():

    u_name = entry.get()
    print(u_name)

    user_details = []
    last_id = ''
    with open('users/users.csv', encoding="utf-8") as users_data:
        users = csv.reader(users_data, delimiter=',')
        for row in users:
            last_id = row[0]

    u_id = int(last_id) + 1

    try:
        search_query = scholarly.search_author(u_name)
        print('Working.............')
        author = next(search_query).fill()
        user_affliation = author.affiliation
        user_key_words = author.interests

        sentence = ''
        for word in user_key_words:
            sentence = sentence + word + ' '

        x = author.publications.__len__()

        if (x < 100):
            num_pub = x
        else:
            num_pub = 100

        abstracts = []
        for count in range(0, num_pub):
            pub = author.publications[count].fill()
            abst = pub.bib.get('abstract')
            abst = str(abst)

            abstracts.append(abst)
            print(abst)

        user_details.append(u_id)
        user_details.append(u_name)
        user_details.append(user_affliation)

        print(user_details)

        # print(user_details)
        preprocess_abstracts(abstracts, u_name, u_id)

    except:
        error()


def preprocess_abstracts(abstracts, u_name, u_id):

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

    predict_interests(preprocessed_abs, u_name, u_id)


# def manual_inputs():
#
#     u_name = 'Sagara Sumathipala'
#     id = '0006'
#
#     abstracts = []
#
#     with open('users/test_user_final_dr_sagara.csv', encoding="utf-8")as file:
#         u_details = csv.reader(file, delimiter=',')
#
#         for detail in u_details:
#             abstracts.append(detail[1])
#
#     preprocess_abstracts(abstracts, u_name, id)


def predict_interests(abstracts, u_name, u_id):

    model_file = 'model/ieee_SVC_kernal=sigmoid_prob.sav'
    # model_file = 'model/ieee_SVC_kernal=rbf_g05_prob.sav'
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
    limit = 0.49

    for p in all_prob:
        # print(p[0][0])
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

    denominator = ml_len + bd_len + dm_len + cv_len + bio_len + ai_len

    ml_score = round(((ml_len / denominator) * 100), 2)
    bd_score = round(((bd_len / denominator) * 100), 2)
    dm_score = round(((dm_len / denominator) * 100), 2)
    cv_score = round(((cv_len / denominator) * 100), 2)
    bio_score = round(((bio_len / denominator) * 100), 2)
    ai_score = round(((ai_len / denominator) * 100), 2)

    print(str(ml_score) + ', ' + str(bd_score) + ', ' + str(dm_score) + ', ' + str(cv_score) + ', ' + str(bio_score)
          + ', ' + str(ai_score))

    with open('users/users.csv', mode='a', encoding="utf-8", newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([u_id, u_name, ml_score, bd_score, dm_score, cv_score, bio_score, ai_score])


# get_user_details(u_name)
# manual_inputs()


def error():

    msg1 = " Ooops... We cann't find you..."
    messagebox.showinfo("Error", msg1)
    msg2 = 'You entered a wrong user name.\n Please Enter the name as same as Google Scholar...'
    messagebox.showinfo("Error", msg2)


top = Tk()

top.geometry('800x500')
top.title('User Login')
top.configure(background='white')
logo = PhotoImage(file='img/logo.png')
photo = Label(top, image=logo, bg='white')
lbl1 = Label(top, bg='white', text='Please enter your Google Scholar Name here :', font=('High Tower Text', 14))
entry = Entry(top, width=30, font=('Comic Sans MS', 14))
button = Button(top, text='Login', width=20, command=lambda: get_user_details())

photo.pack()
lbl1.pack()
entry.pack()
button.pack()

top.mainloop()