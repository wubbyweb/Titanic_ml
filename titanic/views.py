from django.shortcuts import render
from . import fake_model
from . import ml_predict

def home(request):
    return render(request,'index.html')

def result(request):
    user_input_age = int(request.GET["age"])
    prediction = ml_predict.prediction_model(pclass,sex,age,sibsp,parch,fare,embarked,title)
    return render(request,'result.html',{'prediction':prediction})