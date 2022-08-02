# roomvu_task2
binary classification

install all library:

pip install -r requirements.txt

Use following command to start training:

python start_train.py 0|1

0: Continues training

1: Retrain the model

For prediction:

First run the project with:

python manage.py runserver

There are two options for prediction:

1- title and url:  http://127.0.0.1:8000/prediction?title=James Bond No Time To Die At Cinemark 17 Fayetteville GA&url=https://oaklandnewsnow.com/james-bond-no-time-to-die-at-cinemark-17-fayetteville-ga-our-review-of-daniel-craigs-classic/

2- Text: http://127.0.0.1:8000/prediction_text_input?text=James Bond No Time To Die At Cinemark 17 Fayetteville GA
