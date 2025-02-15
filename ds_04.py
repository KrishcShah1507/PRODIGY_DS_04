import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import nltk
nltk.download('vader_lexicon')
def preprocess_text(text):
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'[^\w\s#@]', '', str(text))
    text = text.lower()
    return text
training_file = "C:/Users/DELL/Desktop/ProdigyInfotech_DataScience/4/archive/twitter_training.csv"
df_train = pd.read_csv(training_file)
df_train.columns = ['id', 'topic', 'sentiment', 'text']
df_train['clean_text'] = df_train['text'].apply(preprocess_text)
sid = SentimentIntensityAnalyzer()
df_train['sentiment_compound'] = df_train['clean_text'].apply(lambda x: sid.polarity_scores(x)['compound'])
plt.figure(figsize=(10, 6))
topics = df_train['topic'].unique()
for topic in topics:
    plt.hist(df_train[df_train['topic'] == topic]['sentiment_compound'], bins=30, alpha=0.7, label=topic)
plt.title('Sentiment Analysis by Topic')
plt.xlabel('Compound Sentiment Score')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()