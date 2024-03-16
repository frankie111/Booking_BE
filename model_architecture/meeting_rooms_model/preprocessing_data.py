import pandas as pd
import numpy as np


def try_parsing_date(date_string):
    for fmt in ("%d/%m/%Y", "%d.%m.%Y"):  # Add or modify formats as needed
        try:
            return pd.to_datetime(date_string, format=fmt)
        except ValueError:
            continue
    raise ValueError("no valid date format found")


def preprocess_meeting_room_data(filepath):
    # Load the data
    df = pd.read_csv(filepath)

    df['date'] = df['date'].apply(try_parsing_date)

    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week

    availability_columns = ['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']
    df[availability_columns] = df[availability_columns].applymap(lambda x: 1 if str(x).strip().upper() == 'TRUE' else 0 if str(x).strip().upper() == 'FALSE' else np.nan)

    attendance_columns = ['attendanceNineToEleven', 'attendanceElevenToOne', 'attendanceOneToThree',
                          'attendanceThreeToFive']
    for col in attendance_columns:
        df[col] = df[col].fillna(0)

    return df


if __name__ == "__main__":
    # Load the data
    meeting_rooms = "meeting-rooms.csv"

    meeting_rooms_df = preprocess_meeting_room_data(meeting_rooms)

    meeting_rooms_df.to_csv('preprocessed_working_desks_data1.csv', index=False)