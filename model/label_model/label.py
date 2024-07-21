
import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class label_transformers:
    def __init__(self):
        self.df = pd.read_csv('../../data/products.csv')
        self.features = self.df.columns
        self.clothes_features = self.get_unique_options(self.features[:-1])
        self.weather_features = self.get_unique_options(self.features[-2:-1])
        self.generator = pipeline('text-generation', model='gpt2')

    def get_unique_options(self, features):
        options = {}
        for feature in features:
            options[feature] = self.df[feature].dropna().unique()
        return options

    def generate_description(self, prompt):
        result = self.generator(prompt, max_length=50, num_return_sequences=1)
        return result[0]['generated_text'].strip()

    def compare_feature_options(self, customer_input, feature, options):
        # Create refined prompts for GPT-2 for the specific feature
        prompts = [f"Based on customer preference '{customer_input}', how much do they like the {feature} option '{option}'? Provide a ratio." for option in options]
        
        # Generate descriptions using GPT-2 for the feature
        descriptions = [self.generate_description(prompt) for prompt in prompts]
        print(descriptions)
        # Create embeddings for descriptions
        vectorizer = TfidfVectorizer()
        all_texts = [customer_input] + descriptions
        embeddings = vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity
        similarity_scores = cosine_similarity(embeddings[0:1], embeddings[1:])
        best_match_index = similarity_scores.argmax()
        
        # Return the best match for the feature
        best_match = options[best_match_index]
        return best_match

    def compare_input_to_features(self, customer_input, features):
        results = {}
        for feature, options in features.items():
            # For each feature, find the best match
            best_match = self.compare_feature_options(customer_input, feature, options)
            results[feature] = best_match
        return results

