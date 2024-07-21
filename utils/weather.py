import requests

def get_weather(city):
    api_key = "60ff7a4fd4a74ce889a153225241307"
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Unable to fetch weather data"}
    
