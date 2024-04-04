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


def nano_to_datetime(dt_with_nano):
    """Recreate a datetime object without nanoseconds."""
    return datetime(dt_with_nano.year, dt_with_nano.month, dt_with_nano.day,
                    dt_with_nano.hour, dt_with_nano.minute, dt_with_nano.second,
                    tzinfo=dt_with_nano.tzinfo)
