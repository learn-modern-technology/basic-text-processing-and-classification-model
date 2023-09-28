import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import config
from src.utils import save_file


def custom_vectorizer(i_token_list, y, vector_output_name, i_vector='binary_bag_of_words', i_min_df=5, i_ng_low=1, i_ng_high=3, i_test_size=0.2, i_rs=42):
    """
    Given list of tokens and the dependent variables, the function will
    vectorize the tokens, split the set into train and test and returns the
    data along with the vectorizer
    :param i_token_list: List of processed tokens
    :param y: Dependent variable
    :param i_vector: Vectorizer ('binary_bag_of_words' for count vectors, 'nonbinary_bag_of_words' for binary count vectors, 'ngram' for n-grams and 'term_frequency_inverse_document_frequency' for tf-idf
    :param i_min_df: min_df parameter in CountVectorizer
    :param i_ng_low: Lower value for n-gram
    :param i_ng_high: Higher value for n-gram
    :param i_test_size: Size of test split
    :param i_rs: random seed
    :return: train and test vectors (both X and y), vectorizer
    """
    try:

        # Create vectorizer file path
        vectorized_file = os.path.join(config.output_path, f"{vector_output_name}.pkl")
        # Create vectorizer
        if i_vector == 'nonbinary_bag_of_words':
            p_vectorizer = CountVectorizer(min_df=i_min_df)
        elif i_vector == 'binary_bag_of_words':
            p_vectorizer = CountVectorizer(binary=True, min_df=i_min_df)
        elif i_vector == 'ngram':
            p_vectorizer = CountVectorizer(min_df=i_min_df, ngram_range=(i_ng_low, i_ng_high))
        elif i_vector == 'term_frequency_inverse_document_frequency':
            p_vectorizer = TfidfVectorizer(min_df=i_min_df)
        else:
            raise Exception("Vector has to be one of 'nonbinary_bag_of_words', 'binary_bag_of_words', 'ngram', "
                            "'term_frequency_inverse_document_frequency'")

        # Fit the vectorizer
        p_vectorizer.fit(i_token_list)

        # Fit the vectorizer and tranform the data and assign to X
        X = p_vectorizer.fit_transform(i_token_list)

        # Save the vectorizer
        save_file(vectorized_file, p_vectorizer)

        # Split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i_test_size,
                                                            stratify=y, random_state=i_rs)

    except Exception as e:
        print('An excpetion in custom_vectorizer()-', e)
    else:
        return X_train, X_test, y_train, y_test, p_vectorizer
