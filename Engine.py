import os
import config
import argparse
from src.utils import save_file, read_data
from src.model import custom_vectorizer
from src.processing import process_text
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

pd.options.mode.chained_assignment = None


def train_model(X_train, X_test, y_train, y_test):
    """
    Function to train the model
    :param X_train
    :param X_test
    :param y_train
    :param y_test
    :return: trained model
    """
    try:
        model = LogisticRegression()
        model.fit(X_train, y_train)
        # Make train predictions
        y_train_lr_predicted = model.predict(X_train)
        # Make test predictions
        y_test_lr_predicted = model.predict(X_test)
        # Calculate train accuracy
        train_acc = round(accuracy_score(y_train, y_train_lr_predicted) * 100, 2)
        # Calculate test accuracy
        test_acc = round(accuracy_score(y_test, y_test_lr_predicted) * 100, 2)
        print(f"Train Accuracy: {train_acc}%")
        print(f"Test Accuracy: {test_acc}%")
        print(f'Classification Report -\n{metrics.classification_report(y_test, y_test_lr_predicted)}')
        print(f'Confusion Matrix -\n{metrics.confusion_matrix(y_test, y_test_lr_predicted)}')

    except Exception as e:
        print('Exception in Engine.train_model()', e)
    else:
        return model


def main(input_args):
    try:

        # Create input data file path
        input_file = os.path.join(config.input_path, input_args.file_name)
        print(f'input_file - {input_file}')
        # Read raw data
        data_df = read_data(input_file)
        # Select text and label columns
        tokenized_df = data_df[[config.text_column, config.label_column]]
        # Convert text column to a list of reviews
        # Pre-process the text data
        tokenized_df.loc[:, 'CleanedReview'] = process_text(tokenized_df[config.text_column], config.stem)
        # Create dependent variable
        y = data_df[config.label_column]
        # Vectorize the data and split data into train and test
        X_train, X_test, y_train, y_test, vectorizer = custom_vectorizer(tokenized_df['CleanedReview'],
                                                                         y,
                                                                         input_args.output_name,
                                                                         input_args.vectorizer,
                                                                         config.min_df,
                                                                         config.ng_low,
                                                                         config.ng_high,
                                                                         config.test_size,
                                                                         config.rs)
        # Train the model
        model = train_model(X_train, X_test, y_train, y_test)
        # Create model file path
        model_file = os.path.join(config.output_path, f"{input_args.output_name}_model.pkl")
        # Save the model file
        save_file(model_file, model)

    except Exception as e:
        print('Error in Engine.main()', e)
    else:
        if model:
            print('Model Training Completed!!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="Canva_reviews.xlsx",
                        help="Input file name")
    parser.add_argument("--vectorizer", type=str, default="binary_bag_of_words",
                        help="'nonbinary_bag_of_words', 'binary_bag_of_words', 'ngram', 'term_frequency_inverse_document_frequency'")
    parser.add_argument("--output_name", type=str, default="binary_count_vector",
                        help="Output file name")
    args_main = parser.parse_args()
    main(args_main)
