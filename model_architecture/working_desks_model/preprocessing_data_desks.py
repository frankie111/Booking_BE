import pandas as pd
import numpy as np

def try_parsing_date(date_string):
    for fmt in ("%d/%m/%Y", "%d.%m.%Y"):  # Add or modify formats as needed
        try:
            return pd.to_datetime(date_string, format=fmt)
        except ValueError:
            continue
    raise ValueError("no valid date format found")


def preprocess_working_desks(filepath):
    df = pd.read_csv(filepath)

    df['firstHalf'] = df['firstHalf'].apply(lambda x: 1 if str(x).strip().upper() == 'TRUE' else 0)
    df['secondHalf'] = df['secondHalf'].apply(lambda x: 1 if str(x).strip().upper() == 'TRUE' else 0)

    df['date'] = df['date'].apply(try_parsing_date)

    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week


    return df

if __name__ == "__main__":
    # Load the data
    meeting_rooms = 'hackathon-schema.csv'

    meeting_rooms_df = preprocess_working_desks(meeting_rooms)

    meeting_rooms_df.to_csv('preprocessed_working_desks_data.csv', index=False)
