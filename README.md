# Clothing Recommender System

## Overview and Use Case

The Clothing Recommender System is designed to provide personalized clothing recommendations based on user preferences. By analyzing the style, fabric, pattern, and weather conditions, the system suggests suitable clothing items that align with the user's preferences and current conditions.

### Use Case
- **Personalized Clothing Recommendations:** Users input their style preferences, favorite season, and fabric preferences, and the system provides tailored clothing.
- **Customer Feedback Integration:** Records interactions and feedback to continuously improve recommendations and adapt to user preferences.

### AI Techniques and Tools Used:
- **GPT-2:** Utilized for generating descriptive text based on user preferences and clothing options.
- **TF-IDF Vectorization:** Employed to create embeddings for comparing customer input with clothing options.
- **Cosine Similarity:** Used to measure the similarity between customer input and generated descriptions to find the best match.
- **Random Forest Classifier:** Used for training and predicting clothing recommendations based on historical data.

### Libraries and Frameworks:
- **Pandas:** For data manipulation and handling CSV files.
- **Scikit-learn:** For machine learning tasks, including data splitting, model training, and evaluation.
- **Transformers (Hugging Face):** For text generation with GPT-2.
- **Flask:** For creating the web application and handling user interactions.

## Embedding Methodology

The system uses TF-IDF vectorization to convert text data into numerical vectors. This approach helps in capturing the importance of words in the context of the provided texts, which are then used to compute cosine similarity.

### Steps:
1. **Text Preparation:** Concatenate user input with generated descriptions from GPT-2.
2. **TF-IDF Vectorization:** Transform the text into numerical vectors using TF-IDF.
3. **Cosine Similarity Calculation:** Compute the similarity scores between the user input vector and the vectors of generated descriptions.
4. **Clothing Recommendation:** Use a Random Forest Classifier to predict the best clothing recommendation based on the historical data and user preferences.

## Hugging Face Components

### GPT-2 Text Generation
- **Model:** GPT-2
- **Functionality:** Generates descriptive text based on prompts about user preferences and clothing options.
- **Usage:** `pipeline('text-generation', model='gpt2')` from the Hugging Face Transformers library.

# API Usage

The project includes functionality to fetch current weather data using the WeatherAPI. This information can be used to tailor clothing recommendations based on the current weather conditions in a given city.

### WeatherAPI Integration

We use the `requests` library to make API calls to the WeatherAPI. Below is a brief explanation and example of how the weather data is fetched.
