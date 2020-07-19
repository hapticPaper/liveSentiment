import os, time, pickle
from flask import Flask, jsonify
from flask_restful import reqparse
from flask_cors import CORS
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB


vectorizer_file = os.path.join('model',"tokenizer.sklearn")
tokenizer_file = os.path.join('model',"vectorizer.sklearn")
NBModel = os.path.join('model','simple_sentiment_scoring.sklearn')

vectorizer = pickle.load(open(vectorizer_file, 'rb'))
tokenizer = pickle.load(open(tokenizer_file, 'rb'))
nbModel = pickle.load(open(NBModel, 'rb'))

def predictSentiment(phrases: list):
    """
    Pass a list of one or more phrases, get back their sentiment.
    """
    X = vectorizer.transform(phrases)
    X = tokenizer.transform(X)
    y_predicted = nbModel.predict(X)
    return dict(zip(phrases, y_predicted))


app = Flask('liveSentiment')
CORS(app)


@app.route('/sentiment')
def sentiment():
    parser = reqparse.RequestParser()
    parser.add_argument('phrase', type=str, required=True, help="Provide a phrase to rate", action='append')
    args = parser.parse_args()
    phrase = args['phrase']
    return predictSentiment(phrase)




if __name__=='__main__':
    app.run(host=os.getenv('HOST','0.0.0.0'),
            port=os.getenv('PORT',5000),
            debug=os.getenv('DEBUG',False))


