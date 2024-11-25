from preprocess import *

class Classifier:
    def __init__(self, network):
        self.network = network
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

        for slice in korean_slices:
            slice.show()