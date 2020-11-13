
# importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle
import boto3
from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk
nltk.download('punkt')
embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
from elasticsearch import Elasticsearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

host = 'search-spacy-ner-payzxnqhhfpicotu2zypp56oui.us-east-1.es.amazonaws.com'
awsauth = AWS4Auth('AKIAJ4EJ3QRTAMGGHVZA', 'DhQDCw0UEPtDigzUPGt72miw3FB26xDzqwhy8lNm', 'us-east-1', 'es')
session = boto3.Session(
                    aws_access_key_id='AKIAJ4EJ3QRTAMGGHVZA',
                    aws_secret_access_key='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',
                    region_name='us-east-1')
es = Elasticsearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)


application = Flask(__name__) # initializing a flask app

# app=application
@application.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@application.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:

            #  reading the inputs given by the user
            text=request.form['query']
            s3 = session.resource('s3')
            bucket = s3.Bucket('pdf-converted-docs')
            obj = bucket.Object(key='lambdaout.txt')
            response = obj.get()
            lines = response['Body'].read()
            lines=lines.decode('utf-8')
            doc6 = nltk.tokenize.sent_tokenize(lines)
            corpus_embeddings = embedder.encode(doc6, convert_to_tensor=True)
            #query=request.json["text"]
            #print(query)
            top_k = 5
            query_embedding = embedder.encode(text, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            cos_scores = cos_scores.cpu()
            top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
            b=[]
            for idx in top_results[0:top_k]:
                print(doc6[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
                a=doc6[idx].strip()
                b.append(a)
            r=es.search(index="searchbot", body={"query":{"match":{"attachment.content":{"query":b[0]}}},"_source": False})
            a=r["hits"]["hits"][0]['_id']#loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            # predictions using the loaded model file
            #prediction=loaded_model.predict([[gre_score,toefl_score,university_rating,sop,lor,cgpa,research]])
            #print('prediction is', a)
            # showing the prediction results in a UI
            return render_template('results.html',prediction=a)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	application.run(debug=True) # running the app