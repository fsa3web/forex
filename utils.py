def format_number(number, decimal_places=5):
    """
    Format a number with a specific number of decimal places.
    
    :param number: The number to format
    :param decimal_places: The number of decimal places to show
    :return: A formatted string representation of the number
    """
    return f"{number:.{decimal_places}f}"
