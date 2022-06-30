import pandas as pd
import json
import csv
import nltk
import numpy as np
import pickle
from nltk.corpus import stopwords
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask,render_template,request,url_for

app = Flask(__name__)

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

@app.route("/")
def home():
    with open ("ranveer_show.json","r") as f:
        data = json.load(f)
        name = data['video_urls']
    
    with open ("ranveer.csv",'w') as f:
        fieldname = name[0].keys()
        writer = csv.DictWriter(f,fieldnames=fieldname)
        writer.writeheader()
        for name_ in name:
            writer.writerow(name_)

    df = pd.read_csv("ranveer.csv")

    df1 = df.drop(['video_url','image_url'] , axis = 1 )


    stop = stopwords.words('english')
    stop_words_ = set(stopwords.words('english'))
    wn = WordNetLemmatizer()

    def black_txt(token):
        return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2   
  
    def clean_txt(text):
        clean_text = []
        clean_text2 = []
        text = re.sub("'", "",text)
        text=re.sub("(\\d|\\W)+"," ",text) 
        text = text.replace("nbsp", "")
        clean_text = [ wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
        clean_text2 = [word for word in clean_text if black_txt(word)]
        return " ".join(clean_text2)

    df1['name'] = df1['name'].map(str).apply(clean_txt)
    df1['text'] = df1['text'].map(str).apply(clean_txt)


    tfidf_vectorizer = TfidfVectorizer()

    tfidf_text = tfidf_vectorizer.fit_transform((df1['text'])) #fitting and transforming the vector
    tfidf_name = tfidf_vectorizer.fit_transform((df1['name'])) #fitting and transforming the vector




    return render_template("home.html")


    





#top 5 recommendation
@app.route("/predict",methods=["GET","POST"])

def predict():
    with open ("ranveer_show.json","r") as f:
        data = json.load(f)
        name = data['video_urls']
    
    with open ("ranveer.csv",'w') as f:
        fieldname = name[0].keys()
        writer = csv.DictWriter(f,fieldnames=fieldname)
        writer.writeheader()
        for name_ in name:
            writer.writerow(name_)

    df = pd.read_csv("ranveer.csv")

    df1 = df.drop(['video_url','image_url'] , axis = 1 )


    stop = stopwords.words('english')
    stop_words_ = set(stopwords.words('english'))
    wn = WordNetLemmatizer()

    def black_txt(token):
        return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2   
  
    def clean_txt(text):
        clean_text = []
        clean_text2 = []
        text = re.sub("'", "",text)
        text=re.sub("(\\d|\\W)+"," ",text) 
        text = text.replace("nbsp", "")
        clean_text = [ wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
        clean_text2 = [word for word in clean_text if black_txt(word)]
        return " ".join(clean_text2)

    df1['name'] = df1['name'].map(str).apply(clean_txt)
    df1['text'] = df1['text'].map(str).apply(clean_txt)


    tfidf_vectorizer = TfidfVectorizer()

    tfidf_text = tfidf_vectorizer.fit_transform((df1['text'])) #fitting and transforming the vector
    tfidf_name = tfidf_vectorizer.fit_transform((df1['name'])) #fitting and transforming the vector
    cos_sim = cosine_similarity(tfidf_text,tfidf_text)

    indices = pd.Series(df1.index,index=df1['name']).drop_duplicates()


    def recommendations(title, cos_sim = cos_sim):
        idx = indices[title]
        similarity_scores = list(enumerate(cos_sim[idx]))
        similarity_scores = sorted(similarity_scores,key=lambda x : x[1],reverse=True)

        similarity_scores = similarity_scores[1:6]
    
    #top_5 podcasts recommending based on content of title typed by our user
        pod_indices = [i[0] for i in similarity_scores]
        return pod_indices




    def most_related_videos(recommendations_array):
        #1 element 2D array
        if len(recommendations_array)==1:
            return recommendations_array[0]
        
        else:
        #more than 1 element 2D array
        #adding first ranked videos for several headings
            rank = 0
            empty1 = []
            empty2 = []
            for i in range(0,len(recommendations_array)):
                rank+=1
                empty1.append(recommendations_array[i][0])
          
        #top 5 ranking
            for i in recommendations_array:
                for j in i:
                    no_of_times_repeated = 0
                    for k in range(0,len(recommendations_array)):
                        if j in recommendations_array[k]:
                            no_of_times_repeated+=1
                        if no_of_times_repeated == len(recommendations_array):
                            if j not in empty1:
                                empty1.append(j)
                                rank+=1
            
          
            for i in empty1:
                if i not in empty2:
                    empty2.append(i)
                
       
            for i in range(0,len(recommendations_array)):
                j=1
                if rank==2:
                    if recommendations_array[i][j] not in empty2:
                        empty2.append(recommendations_array[i][1])
                        rank+=1
                if rank==3:
                    if recommendations_array[i][j] not in empty2:
                        empty2.append(recommendations_array[i][1])
                        rank+=1
                if rank==4:
                    if recommendations_array[i][j] not in empty2:
                        empty2.append(recommendations_array[i][1])
                        rank+=1
                j+=1       
                if rank == 5:
                    break
            return empty2
               
          
            
            
            
            
    def recommender_function(word):
        recommendations_array = []
        for i in range(0,len(data['video_urls'])):
            if word in df1['name'][i]:
                recommendations_array.append(recommendations(df1['name'][i]))
        ar_passed = most_related_videos(recommendations_array)
        return ar_passed



    word = request.form['keyword']
    recommendations_array_final  = recommender_function(word)
    #recommended videos
    #passing video name, url, image
    result1 = df['name'].iloc[recommendations_array_final[0]]
    result2 = df['name'].iloc[recommendations_array_final[1]]
    result3 = df['name'].iloc[recommendations_array_final[2]]
    result4 = df['name'].iloc[recommendations_array_final[3]]
    result5 = df['name'].iloc[recommendations_array_final[4]]

    link1 = df['video_url'].iloc[recommendations_array_final[0]]
    link2 = df['video_url'].iloc[recommendations_array_final[1]]
    link3 = df['video_url'].iloc[recommendations_array_final[2]]
    link4 = df['video_url'].iloc[recommendations_array_final[3]]
    link5 = df['video_url'].iloc[recommendations_array_final[4]]
 
    image1 = df['image_url'].iloc[recommendations_array_final[0]]
    image2 = df['image_url'].iloc[recommendations_array_final[1]]
    image3 = df['image_url'].iloc[recommendations_array_final[2]]
    image4 = df['image_url'].iloc[recommendations_array_final[3]]
    image5 = df['image_url'].iloc[recommendations_array_final[4]]
    
    return render_template("result.html",result1 = result1,result2 = result2,result3 = result3,result4= result4,result5=result5,image1=image1,image2=image2,image3=image3,image4=image4,image5=image5,link1=link1,link2=link2,link3=link3,link4=link4,link5=link5,)
    
if __name__ == "__main__":
    app.run(debug=True)

