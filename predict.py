import os
import pandas as pd
import config
import argparse
from src.utils import load_file
from src.processing import process_text
import pandas


def main(input_args):
    """
        Prediction Function
    """
    try:

        # Create model path
        model_file = os.path.join(config.output_path, f"{input_args.model_name}_model.pkl")
        # Load the model
        loaded_model = load_file(model_file)
        # Predictor Input
        text_reviews_df = pd.DataFrame([input_args.text], columns=['review'])
        # Tokenize the input text
        test_clean_review = pd.DataFrame(columns=['CleanedReview'])
        test_clean_review['CleanedReview'] = [process_text(text_reviews_df, 'w')]
        # Create vectorizer path
        vector_file = os.path.join(config.output_path, f"{input_args.model_name}.pkl")
        # Load the vectorizer
        p_vectorizer = load_file(vector_file)
        # Vectorize the tokens
        X_test = p_vectorizer.transform(test_clean_review)
        # Make predictions
        predicted_probability = loaded_model.predict_proba(X_test)[0, 1]*100
        prediction = loaded_model.predict(X_test)
        print(prediction)
        print(f"Probability of Positive Class: {predicted_probability}")
        if prediction == 1:
            print('Positive Sentiment predicted')
        elif prediction == 0:
            print('Negative Setiment is predicted')
        else:
            print('Something fishy!! Please check')
    except Exception as e:
        print('Exception in predictor.main() ', e)
    else:
        print('Prediction Completed!!!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Test review")
    parser.add_argument("--model_name", type=str, default="binary_count_vector",
                        help="Input file name")
    args = parser.parse_args()
    main(args)
