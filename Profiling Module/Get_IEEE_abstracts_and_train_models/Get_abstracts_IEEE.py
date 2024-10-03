import requests
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

url_ml_one = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=author&abstract=Machine+learning'
url_ml_two = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=article_title&abstract=Machine+learning'
url_ml_three = ' http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=publication_title&abstract=Machine+learning'
url_ml_four = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=desc&sort_field=article_number&abstract=Machine+learning'

url_bd_one = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=author&abstract=Big+data'
url_bd_two = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=article_number&abstract=Big+data'
url_bd_three = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=article_title&abstract=Big+data'
url_bd_four = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=publication_title&abstract=Big+data'

url_dm_one = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=author&abstract=Data+Mining'
url_dm_two = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=article_number&abstract=Data+Mining'
url_dm_three = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=article_title&abstract=Data+Mining'
url_dm_four = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=publication_title&abstract=Data+Mining'

url_img_one = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=author&abstract=Computer+Vision'
url_img_two = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=article_number&abstract=Computer+Vision'
url_img_three = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=article_title&abstract=Computer+Vision'
url_img_four = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=publication_title&abstract=Computer+Vision'

url_bioinfor_one = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=author&abstract=Bioinformatics'
url_bioinfor_two = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=article_number&abstract=Bioinformatics'
url_bioinfor_three = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=article_title&abstract=Bioinformatics'
url_bioinfor_four = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=publication_title&abstract=Bioinformatics'

url_ai_one = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=author&abstract=Artificial+Intelligence'
url_ai_two = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=article_number&abstract=Artificial+Intelligence'
url_ai_three = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=article_title&abstract=Artificial+Intelligence'
url_ai_four = 'http://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=zkn22d5qwu42d44tkmmn4sch&format=json&max_records=200&start_record=1&sort_order=asc&sort_field=publication_title&abstract=Artificial+Intelligence'

urls = {url_ai_one, url_ai_two, url_ai_three, url_ai_four}
count = 1

for url in urls:

    response = requests.get(url)
    print(response.status_code)
    data = response.json()
    # print(data)

    bad_chars = ['.', '”', '“', 'ˆ', '/', ':', '-', '(', ')', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ';', '[', ']', ',', '"',
                 '<', '>', '"', '=', '…', 'div', "’", '\'']

    bad_words = ['we', 'the', 'in', 'a', 'during', 'and', 'an', 'this', 'these', 'that', 'paper', 'review', 'more', 'our', 'purpose', 'from',
                 'to', 'ass', 'as', 'public', 'live', 'study', 'along', 'purpose', 'open', 'job', 'main', 'h', 'couple', 'year', 'service',
                 'user', 'new', 'present', 'report', 'use', 'within', 'using', 'work', 'task', 'always']

    for i in data['articles']:
        abstract = i['abstract']

        for chars in bad_chars:
            abstract = abstract.replace(chars, '')

        tokens = nltk.word_tokenize(abstract)

        lower_words = [lw_word.lower() for lw_word in tokens]
        remove_bad_words = [word for word in lower_words if not word in bad_words]

        stop_words = set(stopwords.words('english'))
        with_out_stop_words = [w for w in remove_bad_words if not w in stop_words]

        # stem_abs = [stemmer.stem(word) for word in with_out_stop_words]

        final_abs = ''
        for word in with_out_stop_words:
            final_abs = final_abs + lemmatizer.lemmatize(word) + ' '

        id = 'AI_' + str(count)
        count = count + 1

        lable = '6'

        # Machine Learning = 1
        # Big data = 2
        # Data mining = 3
        # Computer vision = 4
        # Bioinformatics = 5
        # Artificial intelligence = 6

        with open('ieee_dataset_six_areas.csv', mode='a', encoding="utf-8", newline='') as file:

            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([id,final_abs,lable])

            print(id + ' added')


 # ieee_terms = i['index_terms']['ieee_terms']['terms']
    # full_ieee_term = ''
    # for term in ieee_terms:
    #     full_ieee_term = full_ieee_term + term

    # full_author_term = ''
    # if(i['index_terms']['author_terms']['terms'] == True):
    #     author_terms = i['index_terms']['author_terms']['terms']
    #     for au_term in author_terms:
    #         full_author_term = full_author_term + au_term
    # else:
    #     full_author_term = ''