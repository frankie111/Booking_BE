from datetime import datetime, timezone, timedelta


def convert_to_datetime(date_str):
    # Parse the date string into a datetime object
    dt_object = datetime.strptime(date_str, "%d-%m-%Y %H:%M:%S")

    # Define the timezone offset (in hours) for your timezone
    # For example, for Bucharest (UTC+2), the offset is 2 hours
    timezone_offset_hours = 2

    # Create a timezone-aware datetime object with the specified timezone offset
    local_timezone = timezone(timedelta(hours=timezone_offset_hours))
    localized_dt_object = dt_object.replace(tzinfo=local_timezone)

    return localized_dt_object
