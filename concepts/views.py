from django.shortcuts import render
from django.http import HttpResponse
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from .models import Prediction



with open('tfidf.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open("NBclassifier2.pickle", 'rb') as handle:
    classifier = pickle.load(handle)

# Create your views here.
def predict(request):
    if request.method != "POST":
        return render(request,"index.html",{"text": ""})
    else:
        text = request.POST["name"]
        vectorized = tokenizer.transform([text])
        confidance_l = classifier.predict_proba(vectorized)
        predicted_class = classifier.predict(vectorized)[0]
        
        confidance = confidance_l[0][int(predicted_class)]
        pred = Prediction(text=text,Predicted_Class = predicted_class,confidance_score = format(confidance, '.2f'))
        pred.save()
        print(confidance)
        return render(request,"index.html",{"text": f"Predicted Class :{predicted_class},with {format(confidance, '.2f')}% Confidance."})