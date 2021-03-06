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

if __name__ == '__main__':
    app.run(debug=True)
