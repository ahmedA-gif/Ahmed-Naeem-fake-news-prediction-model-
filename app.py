import os
from flask import Flask, request, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Load models
nb_model = joblib.load('nb_model.pkl')
rf_model = joblib.load('rf_model.pkl')
lstm_model = load_model('lstm_model.keras')  # Load LSTM model
tfidf = joblib.load('tfidf_model.pkl')
tokenizer = joblib.load('tokenizer.pkl')  # Load tokenizer for LSTM

# Data cleaning function
def wordrem(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Text preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in text.split() if word not in stop_words]
    text = ' '.join(filtered_words)

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text.split()]
    text = ' '.join(lemmatized_words)

    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in text.split()]
    text = ' '.join(stemmed_words)

    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['news_text']
        model_choice = request.form['model_choice']

        # Clean and preprocess input
        cleaned_input = wordrem(user_input)
        preprocessed_input = preprocess(cleaned_input)

        # Predict based on the selected model
        if model_choice == 'nb':
            vectorized_input = tfidf.transform([preprocessed_input])
            prediction = nb_model.predict(vectorized_input)
            model_name = "Naïve Bayes"
        elif model_choice == 'rf':
            vectorized_input = tfidf.transform([preprocessed_input])
            prediction = rf_model.predict(vectorized_input)
            model_name = "Random Forest"
        elif model_choice == 'lstm':
            # Tokenize and pad input for LSTM
            sequence = tokenizer.texts_to_sequences([preprocessed_input])
            padded_sequence = pad_sequences(sequence, maxlen=100)
            prediction = lstm_model.predict(padded_sequence)
            prediction = np.round(prediction).astype(int)
            model_name = "LSTM"
        else:
            return render_template('index.html', prediction_text="Invalid model selection")

        # Map prediction to "True" or "False"
        result = "True" if prediction[0] == 1 else "False"
        return render_template(
            'index.html',
            user_input=user_input,  # Pass user input back to the template
            prediction_text=f'The news {cleaned_input} is {result} (using {model_name})'
        )

if __name__ == '__main__':
    app.run(debug=True)