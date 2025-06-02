from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

texts = ["I love this", "I hate it", "This is great", "This is awful"]
labels = ["positive", "negative", "positive", "negative"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    x_input = vectorizer.transform([user_input])
    prediction = model.predict(x_input)[0]
    return render_template('index.html', prediction=prediction, text=user_input)

if __name__ == "__main__":
    app.run()
