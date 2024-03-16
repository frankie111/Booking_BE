import pandas as pd

if __name__ == '__main__':


    desk_data = pd.read_csv('preprocessed_working_desks_data.csv')
    meeting_room_data = pd.read_csv('preprocessed_meeting_room_data.csv')

    # Convert date columns to datetime
    desk_data['date'] = pd.to_datetime(desk_data['date'])
    meeting_room_data['date'] = pd.to_datetime(meeting_room_data['date'])

    # Aggregate the working desks dataset by day
    desk_agg = desk_data.groupby([desk_data['date'].dt.date]).agg(
        total_booked_desks_first_half=('firstHalf', 'sum'),
        total_booked_desks_second_half=('secondHalf', 'sum')
    ).reset_index()

    # Adjust the date column to datetime for consistent merging
    desk_agg['date'] = pd.to_datetime(desk_agg['date'])


    # Merge the datasets on the date
    final_dataset = pd.merge(meeting_room_data, desk_agg, how='left', on='date')
    final_dataset.to_csv('merged_data.csv', index=False)

