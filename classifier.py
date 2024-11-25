from preprocess import *

KOREAN_LABEL = "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자배하허호국합육해공외교영준기협정대표"

class Classifier:
    def __init__(self, network, hangul_network):
        self.network = network
        self.hangul_network = hangul_network
        self.plate_image = None
        self.slices = None
        self.license_plate = ""

    def set_image(self, plate_image):
        self.plate_image = plate_image

    def get_license_plate(self):
        if self.plate_image:
            self.slices = preprocess(self.plate_image)
        else:
            raise ValueError("There isn't plate image")
        
        number_slices = self.slices[:3] + self.slices[-4:]
        korean_slices = self.slices[3:-4]

        for slice in number_slices:
            slice_array = np.array(slice)
            slice_array = slice_array[np.newaxis, np.newaxis, :, :]
            slice_array = slice_array.astype(np.float32)

            scores = self.network.predict(slice_array)
            predicted_label = np.argmax(scores)
            self.license_plate += str(predicted_label)

        korean_character = merge_korean(korean_slices)
        korean_character.show()
        korean_array = np.array(korean_character)
        korean_array = korean_array[np.newaxis, np.newaxis, :, :]
        korean_array = korean_array.astype(np.float32)

        scores = self.hangul_network.predict(korean_array)
        predicted_label = np.argmax(scores)
        predicted_label = KOREAN_LABEL[predicted_label]
        self.license_plate = self.license_plate[:3] + predicted_label + self.license_plate[3:]

        print(self.license_plate)