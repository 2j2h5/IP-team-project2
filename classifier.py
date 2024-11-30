from preprocess import *

KOREAN_LABEL = "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자배하허호국합육해공외교영준기협정대표"

class Classifier:
    def __init__(self, network, hangul_network):
        self.network = network
        self.hangul_network = hangul_network
        self.plate_image = None
        self.slices = None
        self.license_plate = ""
        self.korean_offset = 0

    def set_image(self, plate_image):
        self.plate_image = plate_image
        self.slices = None
        self.license_plate = ""
        self.korean_offset = 0

    def get_license_plate(self, threshold=128): # get license from input image
        if self.plate_image:
            self.slices = preprocess(self.plate_image, threshold)
        else:
            raise ValueError("There is no plate image")
      
        if len(self.slices)==9:
            number_slices = self.slices[:3] + self.slices[-4:]
            korean_slices = self.slices[3:-4]
            self.korean_offset = 3
        elif len(self.slices)==7:
            number_slices = self.slices[:2] + self.slices[-4:]
            korean_slices = self.slices[2:-4]
            self.korean_offset = 2
        else:
            determinant = self.slices[1:3]
            if (determinant[0].width - determinant[1].width) > determinant[0].width/5:
                number_slices = self.slices[:2] + self.slices[-4:]
                korean_slices = self.slices[2:-4]
                self.korean_offset = 2
            else:
                number_slices = self.slices[:3] + self.slices[-4:]
                korean_slices = self.slices[3:-4]
                self.korean_offset = 3

        for slice in number_slices:
            slice_array = np.array(slice)
            slice_array = slice_array[np.newaxis, np.newaxis, :, :]
            slice_array = slice_array.astype(np.float32)

            scores = self.network.predict(slice_array)
            predicted_label = np.argmax(scores)
            self.license_plate += str(predicted_label)

        korean_character = merge_korean(korean_slices)
        korean_array = np.array(korean_character)
        korean_array = korean_array[np.newaxis, np.newaxis, :, :]
        korean_array = korean_array.astype(np.float32)

        scores = self.hangul_network.predict(korean_array)
        predicted_label = np.argmax(scores)
        predicted_label = KOREAN_LABEL[predicted_label]
        self.license_plate = self.license_plate[:self.korean_offset] + predicted_label + self.license_plate[self.korean_offset:]

        return self.license_plate

    def parse_license_plate(self):
        license_plate = self.license_plate
        # Separate numbers and Korean character from the license plate
        numbers_part = license_plate[:self.korean_offset]  # First 3 digits
        korean_char = license_plate[self.korean_offset]    # One Korean character

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