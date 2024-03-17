import pickle
import pandas as pd

def load_model(model_filepath):
    with open(model_filepath, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_input(date, room):
    # Assuming date is in 'YYYY-MM-DD' format
    # Convert date to datetime
    date = pd.to_datetime(date)
    # Extract features from the date
    day_of_week = date.dayofweek
    month = date.month
    week_of_year = date.isocalendar()[1] # ISO week of the year
    # Convert room to categorical code
    room_code = room_map[room]
    # Create input array
    input_data = [[day_of_week, month, week_of_year, room_code]]
    return input_data

def predict_availability(model, input_data):
    prediction = model.predict(input_data)
    availability = "Available" if prediction == 1 else "Not Available"
    return availability

# Load the trained model
model = load_model('model_test_fewer_features.pkl')

# Mapping room names to categorical codes (assuming this was done during preprocessing)
room_map = {
    "Pit-Lane": 0,
    "Dry-lane": 1,
    "Joker Lap": 2,
    "Quick 8": 3,
    "Pole Position": 4,
    "Cockpit": 5
}

def main():

    # Example input
    date = '2024-12-02'
    room = 'Pit-Lane'

    # Preprocess the input data
    input_data = preprocess_input(date, room)

    # Predict room availability
    availability = predict_availability(model, input_data)

    print(f"The room '{room}' on {date} is {availability}")
main()