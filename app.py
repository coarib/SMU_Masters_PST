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


@app.route('/translate', methods=['GET'])
def translate():
    # Params for model are in here!
    args = request.args.to_dict()

    #button_id = request.form['submit_button']
    #print(button_id)
	
    button_id=''
    button_id = args['submit_button']
    print(button_id)
    #set input values
    sent = args['fromText']
    from_language=args['fromLang']
    to_language=args['toLang']
    #google_text=args['googleText']
    #pst_text=args['pstText']
	
    google_text=''
    pst_text=''
	
    if (button_id=="usePSTTranslation") :
        updateTranslations(from_language,sent,to_language,pst_text)
		
    if (button_id=="useGoogleTranslation") :
        updateTranslations(from_language,sent,to_language,google_text)
	
    import numpy as np
    import pickle
    import sklearn
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import euclidean_distances

    #set input values
    sent = args['fromText']
    from_language=args['fromLang']
    to_language=args['toLang']
    
    # read python df of languages matching the "from_language" from the appropriate smart-named pkl file
    pkl_file = open('sentences_'+from_language+'.pkl', 'rb')
    from_language_sentences = pickle.load(pkl_file)
    pkl_file.close()

    #create a list of the values to be parsed, using our translatable sentence as index 0
    sent=sent.strip().lower().replace('(', '').replace(')', '')

    corpus = [] 
    corpus.append(sent)

    #Append the appropriate language values
    corpus= corpus + from_language_sentences['text'].tolist()

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    #print (tfidf_matrix.shape)
    
    #grab the closest match of the sentence based on cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    c_similarity = []
    c_similarity.append(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]))

    #grab the closest match of the sentence based on cosine similarity
    index_max = np.argmax(c_similarity)
    #print("The closest sentence match is: " + corpus[index_max+1])

    #grab the matching sentence ID for the language using the identified text and convert it to a string
    sentence_id = from_language_sentences.loc[from_language_sentences['text'] == corpus[index_max+1]]['sentence_id'].values
    sentence_id = int(sentence_id)
    percent_match = str(round(np.amax(c_similarity),4)*100)
    #print("That sentence is sentence ID: " + str(sentence_id))
    #print("The value of the cosine similarity for the sentences is: " + str(np.amax(c_similarity)))
        
    # read translations pandas df back from the pkl file
    pkl_file = open('translations.pkl', 'rb')
    translations_df = pickle.load(pkl_file)
    pkl_file.close()

    #print(translations_df.head())
    
    #print(translations_df.loc[(translations_df['output_language_key'] == to_language)])
    #print(translations_df.loc[(translations_df['output_language_key'] == to_language) & (translations_df['input_sentence_id'] == sentence_id)])
            
    #get the resulting translations for the input sentence and desired language
    translations = translations_df.loc[(translations_df['output_language_key'] == to_language) & (translations_df['input_sentence_id'] == sentence_id)]
    
	#print(translations['input_text'].values[0])
    #print(translations['output_text'].values[0])
	
    #Get Google Results
    from googletrans import Translator
    translator = Translator()
    translated = translator.translate(sent, src=from_language, dest=to_language)
    
    return render_template('translated.html', googleResult = translated.text, resultFromText=translations['input_text'].values[0], resultToText=translations['output_text'].values[0], percentMatch=percent_match, resultOrigText = args['fromText'], resultFromLang=args['fromLang'], resultToLang=args['toLang'])



def updateTranslations(from_language, new_from_text, to_language, new_to_text):
    print('in updateTranslations')
	
    import mysql.connector
    import numpy as np
    import pandas as pd
    import pyodbc
    import pickle
    import sqlalchemy as sql
    from sqlalchemy import create_engine
    import re
	
	#The purpose of this code block is to find the test if our sentences are in the database

    #sanitize sentences
    new_from_text = re.sub('[!@#$%;:.?()"\'^\{\}\[\]|\\\/<>=`~*&]', '', new_from_text).strip().lower()
    new_to_text = re.sub('[!@#$%;:.?()"\'^\{\}\[\]|\\\/<>=`~*&]', '', new_to_text).strip().lower()


    # ====== Connection ====== #
    # Connecting to mysql by providing a sqlachemy engine
    engine = create_engine('mysql+pymysql://root:MBBmasters!@35.232.80.174/masters', echo=False)

    #Check for new from_text and insert if it is new
    df = pd.read_sql("select sentence_id from Sentences where language_key \
    ='"+from_language+"' and text = '"+ new_from_text +"'", engine)

    if len(df.index) == 0:
        #Insert due to no entries
        data = [[new_from_text, from_language]] 
        new_sentence_insert = pd.DataFrame(data, columns = ['text', 'language_key']) 
        new_sentence_insert.to_sql('Sentences', con=engine, if_exists='append', index = False)
        #Requery to get the value
        df = pd.read_sql("select sentence_id from Sentences where language_key \
        ='"+from_language+"' and text = '"+ new_from_text +"'", engine)
    
    from_sentence_id = df['sentence_id'].values[0]
    
    #Check for new to_text and insert if it is new
    df = pd.read_sql("select sentence_id from Sentences where language_key \
    ='"+to_language+"' and text = '"+ new_to_text +"'", engine)

    if len(df.index) == 0:
        #Insert due to no entries
        data = [[new_to_text, to_language]] 
        new_sentence_insert = pd.DataFrame(data, columns = ['text', 'language_key']) 
        new_sentence_insert.to_sql('Sentences', con=engine, if_exists='append', index = False)
        #Requery to get the value
        df = pd.read_sql("select sentence_id from Sentences where language_key \
        ='"+to_language+"' and text = '"+ new_to_text +"'", engine)

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
	
    return 

if __name__ == '__main__':
    app.run(debug=True)
