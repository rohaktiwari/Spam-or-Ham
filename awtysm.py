import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string
import joblib

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['message'] = df['message'].apply(preprocess_text)
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

y_pred = clf.predict(X_test_counts)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

joblib.dump(clf, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'count_vectorizer.pkl')

def classify_email(email):
    email = preprocess_text(email)
    email_counts = vectorizer.transform([email])
    prediction = clf.predict(email_counts)
    return prediction[0]

#testingnew_email = "Congratulations! You've won a free ticket to the Bahamas. Claim now!"
#print(classify_email(new_email))