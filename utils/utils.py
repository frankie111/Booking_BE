from datetime import datetime


def convert_to_timestamp(date_str, format='%d-%m-%Y %H:%M:%S'):
    """
    Convert a date/time string to a Unix timestamp.

    Args:
        date_str (str): The date/time string.
        format (str): The format of the date/time string. Default is '%Y-%m-%d %H:%M:%S'.

    Returns:
        int: The Unix timestamp corresponding to the input date/time.
    """
    # Parse the date/time string using the specified format
    dt_obj = datetime.strptime(date_str, format)

    # Convert the datetime object to a timestamp
    timestamp = int(dt_obj.timestamp())

    return timestamp
