import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

class clothing_recommender:
    def __init__(self):
        self.data_path = '../../data/products.csv'
        self.df = None
        self.label_encoders = {}
        self.le_target = LabelEncoder()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
    
    def encode_features(self):
        for column in self.df.columns[:-1]:  # Skip the last column (recommended_clothing_items)
            le = LabelEncoder()
            self.df[column] = le.fit_transform(self.df[column])
            self.label_encoders[column] = le
        
        self.df['recommended_clothing_items'] = self.le_target.fit_transform(self.df['recommended_clothing_items'])
    
    def split_data(self):
        X = self.df[self.df.columns[:-1]]  # All columns except the last one
        y = self.df['recommended_clothing_items']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        unique_labels = list(set(y_test) | set(y_pred))
        filtered_target_names = [self.le_target.classes_[label] for label in unique_labels]
        print(classification_report(y_test, y_pred, labels=unique_labels, target_names=filtered_target_names))
    
    def predict_recommendation(self, input_data):
        encoded_input = []
        for feature in input_data:
            encoded_feature = self.label_encoders[feature].transform([input_data[feature]])[0]
            encoded_input.append(encoded_feature)
        
        prediction = self.model.predict([encoded_input])
        return self.le_target.inverse_transform(prediction)[0]
    
    def save_model(self, model_path, encoders_path):
        # Ensure paths are correct and exist
        joblib.dump(self.model, model_path + 'model.pkl')
        joblib.dump(self.le_target, encoders_path + 'le_target.pkl')
        joblib.dump(self.label_encoders, encoders_path + 'label_encoders.pkl')
    
    def load_model(self, model_path, encoders_path):
        # Combine paths correctly
        self.model = joblib.load(model_path + 'model.pkl')
        self.le_target = joblib.load(encoders_path + 'le_target.pkl')
        self.label_encoders = joblib.load(encoders_path + 'label_encoders.pkl')