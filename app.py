"""
Starts Recommender Web Server
"""

from flask import Flask
from flask import Flask, abort, request, render_template
from flask import Response
import logging
import glob
import inflection


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


def populate_main(my_return_df):

    import numpy as np
    import pickle
    import sklearn
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import euclidean_distances
    import re
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
	
    # read python df of languages matching the "from_language" from the appropriate smart-named pkl file
    pkl_file = open('sentences_'+my_return_df['from_language'].values[0]+'.pkl', 'rb')
    from_language_sentences = pickle.load(pkl_file)
    pkl_file.close()

    #Clean input
    my_return_df.ix[0, 'sent'] = re.sub('[!@#$%;:.?()"\'^\{\}\[\]|\\\/<>=`~*&]', '', my_return_df['orig_sent'].values[0]).strip().lower()
	

    #create a list of the values to be parsed, using our translatable sentence as index 0
    corpus = [] 
    corpus.append(my_return_df['sent'].values[0])

    #Append the appropriate language values
    corpus= corpus + from_language_sentences['text'].tolist()

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    #grab the closest match of the sentence based on cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    c_similarity = []
    c_similarity.append(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]))

    #grab the closest match of the sentence based on cosine similarity
    index_max = np.argmax(c_similarity)
  
    #grab the matching sentence ID for the language using the identified text and convert it to a string
    sentence_id = from_language_sentences.loc[from_language_sentences['text'] == corpus[index_max+1]]['sentence_id'].values
    sentence_id = int(sentence_id)
    percent_match = str(round(np.amax(c_similarity),4)*100)
    my_return_df.ix[0, 'percent_match'] = percent_match
	
    # read translations pandas df back from the pkl file
    pkl_file = open('translations.pkl', 'rb')
    translations_df = pickle.load(pkl_file)
    pkl_file.close()

    #get the resulting translations for the input sentence and desired language
    translations = translations_df.loc[(translations_df['output_language_key'] == my_return_df['to_language'].values[0]) & (translations_df['input_sentence_id'] == sentence_id)]
    
    my_return_df.ix[0, 'sent'] = translations['input_text'].values[0]
    my_return_df.ix[0, 'pst_text'] = translations['output_text'].values[0]
    
    #Get Google Results
    from googletrans import Translator
    translator = Translator()
    translated = translator.translate(my_return_df['orig_sent'].values[0], src=my_return_df['from_language'].values[0], dest=my_return_df['to_language'].values[0])
    my_return_df.ix[0, 'google_text'] = translated.text
	
	
    return




@app.route('/translate', methods=['GET'])
def translate():
    import pandas as pd
	
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    # Params for model are in here!
    args = request.args.to_dict()

    button_id=''
    button_id = args['submit_button']
    print(button_id)
	
    #set input values
    sent = args['fromText']
    from_language=args['fromLang']
    to_language=args['toLang']
    google_text=args['googleText']
    pst_text=args['pstText']
	
    my_return_obj = {'orig_sent': [sent], 'sent': [sent], 'from_language': [from_language], 'to_language': [to_language], 'google_text': [google_text], 'pst_text': [pst_text], 'percent_match': ["0"]}
    my_return_df = pd.DataFrame(data=my_return_obj)
	
    if (button_id=="usePSTTranslation") :
        updateTranslations(my_return_df,pst_text)
		
    if (button_id=="useGoogleTranslation") :
        updateTranslations(my_return_df,google_text)
	
    populate_main(my_return_df)
		
    print("orig_sent: " + my_return_df['orig_sent'].values[0])
    print("sent: " + my_return_df['sent'].values[0])
    print("from_language: " + my_return_df['from_language'].values[0])
    print("to_language: " + my_return_df['to_language'].values[0])
    print("google_text: " + my_return_df['google_text'].values[0])
    print("pst_text: " + my_return_df['pst_text'].values[0])
    print("percent_match: " + my_return_df['percent_match'].values[0])
	
    return render_template('translated.html', googleResult = my_return_df['google_text'].values[0], resultFromText=my_return_df['sent'].values[0], resultToText=my_return_df['pst_text'].values[0], percentMatch=my_return_df['percent_match'].values[0], resultOrigText = my_return_df['orig_sent'].values[0], resultFromLang=my_return_df['from_language'].values[0], resultToLang=my_return_df['to_language'].values[0])

	

def updatePklFiles():
    import mysql.connector
    import numpy as np
    import pandas as pd
    import pickle
    import sqlalchemy as sql
    from sqlalchemy import create_engine
    import os
    import re
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    # ====== Connection ====== #
    # Connecting to mysql by providing a sqlachemy engine
    engine = create_engine('mysql+pymysql://root:' + os.environ['pst_pwd'] + '@35.232.80.174/masters', echo=False)

    #Export English sentences in a pkl file (EN=English)
    df = pd.read_sql("select text, sentence_id from Sentences where language_key='EN'", engine)
    output = open('sentences_EN.pkl', 'wb')
    pickle.dump(df, output)
    output.close()

    #Export Portuguese sentences in a pkl file (PT=Portuguese)
    df = pd.read_sql("select text, sentence_id from Sentences where language_key='PT'", engine)
    output = open('sentences_PT.pkl', 'wb')
    pickle.dump(df, output)
    output.close()

    #Export Hindi sentences in a pkl file (HI=Hindi)
    df = pd.read_sql("select text, sentence_id from Sentences where language_key='HI'", engine)
    output = open('sentences_HI.pkl', 'wb')
    pickle.dump(df, output)
    output.close()

    #Export Translations sentences in a pkl file 
    df = pd.read_sql("SELECT a.sentence_id as input_sentence_id, \
    a.language_key as input_language_key, a.text as input_text, \
    b.text as output_text, b.sentence_id as output_sentence_id, \
    b.language_key as output_language_key FROM masters.Translations \
    left join masters.Sentences a on a.sentence_id=Translations.sentence_id_1 \
    left join masters.Sentences b on b.sentence_id=Translations.sentence_id_2", engine)
    output = open('translations.pkl', 'wb')
    pickle.dump(df, output)
    output.close()

    return

def updateTranslations(my_return_df, new_to_text):
    print('in updateTranslations')
	
    import mysql.connector
    import numpy as np
    import pandas as pd
    import pyodbc
    import pickle
    import sqlalchemy as sql
    from sqlalchemy import create_engine
    import re
    import os
	
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
	
	
    print("orig_sent: " + my_return_df['orig_sent'].values[0])
    print("sent: " + my_return_df['sent'].values[0])
    print("from_language: " + my_return_df['from_language'].values[0])
    print("to_language: " + my_return_df['to_language'].values[0])
    print("google_text: " + my_return_df['google_text'].values[0])
    print("pst_text: " + my_return_df['pst_text'].values[0])
    print("percent_match: " + my_return_df['percent_match'].values[0])
    print("new_to_text: " + new_to_text)
	
	#The purpose of this code block is to find the test if our sentences are in the database

    #sanitize sentences
    new_to_text = re.sub('[!@#$%;:.?()"\'^\{\}\[\]|\\\/<>=`~*&]', '', new_to_text).strip().lower()
    my_return_df.ix[0, 'sent'] = re.sub('[!@#$%;:.?()"\'^\{\}\[\]|\\\/<>=`~*&]', '', my_return_df['sent'].values[0]).strip().lower()

    # ====== Connection ====== #
    # Connecting to mysql by providing a sqlachemy engine
    engine = create_engine('mysql+pymysql://root:' + os.environ['pst_pwd'] + '@35.232.80.174/masters', echo=False)

    #Check for new from_text and insert if it is new
    df = pd.read_sql("select sentence_id from Sentences where language_key \
    ='"+my_return_df['from_language'].values[0]+"' and text = '"+ my_return_df['sent'].values[0] +"'", engine)

    if len(df.index) == 0:
        #Insert due to no entries
        data = [[my_return_df['sent'].values[0], my_return_df['from_language'].values[0]]] 
        new_sentence_insert = pd.DataFrame(data, columns = ['text', 'language_key']) 
        new_sentence_insert.to_sql('Sentences', con=engine, if_exists='append', index = False)
        #Requery to get the value
        df = pd.read_sql("select sentence_id from Sentences where language_key \
        ='"+my_return_df['from_language'].values[0]+"' and text = '"+ my_return_df['sent'].values[0] +"'", engine)
    
    from_sentence_id = df['sentence_id'].values[0]
    
    #Check for new to_text and insert if it is new
    df = pd.read_sql("select sentence_id from Sentences where language_key \
    ='"+my_return_df['to_language'].values[0]+"' and text = '"+ new_to_text +"'", engine)

    if len(df.index) == 0:
        #Insert due to no entries
        data = [[new_to_text, my_return_df['to_language'].values[0]]] 
        new_sentence_insert = pd.DataFrame(data, columns = ['text', 'language_key']) 
        new_sentence_insert.to_sql('Sentences', con=engine, if_exists='append', index = False)
        #Requery to get the value
        df = pd.read_sql("select sentence_id from Sentences where language_key \
        ='"+my_return_df['to_language'].values[0]+"' and text = '"+ new_to_text +"'", engine)

    to_sentence_id = df['sentence_id'].values[0]
	
	#The goal of this text block is to insert the bidirectional translation if it is a new string

    df = pd.read_sql("select translation_id from Translations where \
    sentence_id_1 ="+str(from_sentence_id)+" and sentence_id_2 = "+ str(to_sentence_id), engine)

    if len(df.index) == 0:
        data = [[from_sentence_id, to_sentence_id]] 
        new_translation_insert = pd.DataFrame(data, columns = ['sentence_id_1', 'sentence_id_2']) 
        new_translation_insert.to_sql('Translations', con=engine, if_exists='append', index = False)
        df = pd.read_sql("select translation_id from Translations where \
        sentence_id_1 ="+str(from_sentence_id)+" and sentence_id_2 = "+ str(to_sentence_id), engine)

    translation_1_df = pd.read_sql("select * from Translations where \
    sentence_id_1 ="+str(from_sentence_id)+" and sentence_id_2 = "+ str(to_sentence_id), engine)

    #print(translation_1_df)

    df = pd.read_sql("select translation_id from Translations where \
    sentence_id_2 ="+str(from_sentence_id)+" and sentence_id_1 = "+ str(to_sentence_id), engine)

    if len(df.index) == 0:
        data = [[from_sentence_id, to_sentence_id]] 
        new_translation_insert = pd.DataFrame(data, columns = ['sentence_id_2', 'sentence_id_1']) 
        new_translation_insert.to_sql('Translations', con=engine, if_exists='append', index = False)
        df = pd.read_sql("select translation_id from Translations where \
        sentence_id_2 ="+str(from_sentence_id)+" and sentence_id_1 = "+ str(to_sentence_id), engine)

    translation_2_df = pd.read_sql("select * from Translations where \
    sentence_id_2 ="+str(from_sentence_id)+" and sentence_id_1 = "+ str(to_sentence_id), engine)

    #print(translation_2_df)
	
    updatePklFiles()
	
    return 

if __name__ == '__main__':
    app.run(debug=True)
