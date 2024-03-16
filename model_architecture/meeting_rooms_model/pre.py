import pandas as pd

df = pd.read_csv('preprocessed_meeting_room_data1.csv')

df['day_of_week'] = df['day_of_week'].astype('category')

