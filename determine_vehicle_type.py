def parse_license_plate(license_plate):
    # Separate numbers and Korean character from the license plate
    numbers_part = license_plate[:3]  # First 3 digits
    korean_char = license_plate[3]    # One Korean character

    # Initialize variables
    vehicleBodyType = None
    usage = None
    isCommercial = None

    # Handle number plate conditions
    try:
        numbers_part = int(numbers_part)  
    except ValueError:
        print("Error: License plate format is incorrect.")
        return None

    # Determine vehicle type and usage
    if 100 <= numbers_part <= 699 or 1 <= numbers_part <= 69:
        vehicleBodyType = "Passenger Car"
        if 100 <= numbers_part <= 699:
            if korean_char in "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주":
                usage = "Non-commercial (private)"
                isCommercial = False
            elif korean_char in "허하호":
                usage = "Rental Business"
                isCommercial = True
        elif 1 <= numbers_part <= 69:
            if korean_char in "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주":
                usage = "Non-commercial (private) electric car"
                isCommercial = False
            elif korean_char in "바사아자배":
                usage = "Commercial (taxi)"
                isCommercial = True

    elif 700 <= numbers_part <= 799:
        vehicleBodyType = "Bus"
        if korean_char in "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주":
            usage = "Non-commercial (private)"
            isCommercial = False
        elif korean_char in "허하호":
            usage = "Rental Business"
            isCommercial = True
        elif korean_char in "바사아자배":
            usage = "Commercial (bus)"
            isCommercial = True

    elif 800 <= numbers_part <= 979:
        vehicleBodyType = "Truck"
        if korean_char in "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주":
            usage = "Non-commercial (private)"
            isCommercial = False
        elif korean_char in "바사아자배":
            usage = "Commercial (truck)"
            isCommercial = True

    elif 980 <= numbers_part <= 997:
        vehicleBodyType = "Special Vehicle"
        if korean_char in "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주":
            usage = "Non-commercial"
            isCommercial = False
        elif korean_char in "바사아자배":
            usage = "Commercial (business use)"
            isCommercial = True

    elif 998 <= numbers_part <= 999:
        vehicleBodyType = "Emergency Vehicle"
        usage = "Police or Fire Truck"
        isCommercial = False

    # Return result as a dictionary
    return {
        "vehicleBodyType": vehicleBodyType,
        "usage": usage,
        "isCommercial": isCommercial,
    }
