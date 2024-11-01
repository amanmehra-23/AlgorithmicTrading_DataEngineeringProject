import nltk

# Set the NLTK data path
nltk.data.path.append('/Users/amanmehra/Desktop/nltk_data')  # replace with your actual path
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()


def get_sentiment(text):
    scores = sia.polarity_scores(text)
    return scores['compound']
