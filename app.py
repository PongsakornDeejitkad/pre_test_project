from flask import Flask, render_template, request
from model.training_model.predictive_model.clothing_recommender import clothing_recommender
from model.label_model.label import label_transformers
from utils.weather import get_weather

app = Flask(__name__)

def predict(customer_input,location):
    
    label_transformed = label_transformers()
    recommender = clothing_recommender()
    
    weather_data = get_weather(location)
    current_weather = weather_data['current']['condition']['text']

    transformed_weather = label_transformed.compare_input_to_features(current_weather, label_transformed.weather_features)
    transformed_input = label_transformed.compare_input_to_features(customer_input, label_transformed.clothes_features)
    merged_dict = transformed_input | transformed_weather

    
    # Load the model and encoders
    recommender.load_model(model_path='model/training_model/best_param/', encoders_path='model/training_model/encoders/')
    predicted_item = recommender.predict_recommendation(merged_dict)
    return predicted_item,current_weather




@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        customer_input = request.form['customer_input']
        location = request.form['location']
        
        # Call predict with both parameters
        recommended_product,current_weather = predict(customer_input,location)

        return render_template('index.html', 
                               recommended_product=recommended_product,
                               customer_input=customer_input,
                               current_weather=current_weather,
                               location=location)
    return render_template('index.html', 
                           recommended_product=None,
                           customer_input=None,
                           location=None)

if __name__ == '__main__':
    app.run(debug=True)
