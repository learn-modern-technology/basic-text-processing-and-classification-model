import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer

stop_words = stopwords.words('english')

# Custom Stoplist
custome_stopwords = ["hey!", "it!", "you!", "i'm", "i", "i've", "me", "my", "myself", "we", "our", "ours", "ourselves",
                     "you", "you're", "you've", "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he",
                     "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it", "it's", "its", "itself",
                     "they", "them", "their",
                     "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "that'll", "these",
                     "those", "am", "is", "are",
                     "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing",
                     "a", "an", "the", "and",
                     "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
                     "against", "between", "into",
                     "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
                     "on", "off", "over",
                     "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "all", "any",
                     "both", "each",
                     "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
                     "than", "too", "very",
                     "s", "t", "can", "will", "just", "don", "don't", "should", "should've", "now", "d", "ll", "m", "o",
                     "re", "ve", "y", "ain",
                     "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
                     "needn",
                     "shan", "shan't", "shouldn", "wasn", "weren", "won", "rt", "rt", "qt", "for", "the", "with", "in",
                     "of", "and",
                     "its", "it", "this", "i", "have", "has", "would", "could", "you", "a", "an", "be", "am", "can",
                     "will", "to", "on", "is",
                     "by", "ive", "im", "your", "we", "are", "at", "as", "any", "ebay", "thank", "hello", "know",
                     "need", "want",
                     "look", "hi", "sorry", "http", "https", "body", "dear", "hello", "hi", "thanks", "sir", "tomorrow",
                     "sent", "send", "see",
                     "there", "welcome", "what", "well", "us"]

stop_words.extend(custome_stopwords)


def process_text(review_df, stem='w'):
    """
    Given a text, the function converts the text into lower case,
    removes stopwords, removes punctuations, performs stemming and
    returns the processed text
    :param review_df: raw review in text
    :param stem: Stemmer - 'p' for PorterStemmer and 'l' for
                        LancasterStemmer
    :return: processed text
    """
    # Remove stopwords
    review_df['CleanedReview'] = review_df['review'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words))
    # Remove punctuation
    review_df['CleanedReview'] = review_df['CleanedReview'].apply(lambda x: remove_punctuations(x))

    # Create stemmer
    if stem == 'p':
        porter_stemmer = PorterStemmer()
        review_df['CleanedReview'] = review_df['CleanedReview'].apply(lambda x: ' '.join(porter_stemmer.stem(word) for word in x.split()))
    elif stem == 'l':
        lancaster_stemmer = LancasterStemmer()
        review_df['CleanedReview'] = review_df['CleanedReview'].apply(lambda x: ' '.join(lancaster_stemmer.stem(word) for word in x.split()))
    elif stem == 'w':
        stemmer = WordNetLemmatizer()
        review_df['CleanedReview'] = review_df['CleanedReview'].apply(lambda x: ' '.join(stemmer.lemmatize(word) for word in x.split()))
    else:
        raise Exception("stem has to be either 'p' for Porter or 'l' for Lancaster or 'w' for WordNetLemmatizer ")

    # Return clean string
    return review_df


# Function to remove punctuation in the text
def remove_punctuations(text):
    """
        text: a string    
        return: modified initial string
    """
    # text = text.replace("\d+"," ") #removing digits
    text = re.sub(r"(?:\@|https?\://)\S+", '', text)  # removing mentions and urls
    text = text.lower()
    text = re.sub('[0-9]+', '', text)  # removing numeric characters
    text = re.sub('[/(){}\[\]\|@,;!]', ' ', text)  # replace symbols by space in text
    text = re.sub('[^0-9a-z #+_]', ' ', text)  # replace symbols which are in BAD_SYMBOLS_RE from text
    text = text.strip()
    return text
