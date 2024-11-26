from PIL import Image
from simple_conv_net import SimpleConvNet
from hangul_conv_net import HangulConvNet
from classifier import Classifier

#prepare SimpleConvNet
network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
network.load_params("params.pkl")

#prepare HangulConvNet
hangul_network = HangulConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=54, weight_init_std=0.01)
hangul_network.load_params("hangul-params.pkl")

#prepare image
image_path = 'images/number-plate/number-plate-clean-1.jpg'
image_path2 = 'images/number-plate/number-plate-clean-2.jpg'
image_path3 = 'images/number-plate/number-plate-clean-3.jpg'

plate_image = Image.open(image_path).convert('L')
plate_image2 = Image.open(image_path2).convert('L')
plate_image3 = Image.open(image_path3).convert('L')

#do classification
classifier = Classifier(network, hangul_network)

classifier.set_image(plate_image)
license_number = classifier.get_license_plate()
print(license_number)
vehicle_type = classifier.parse_license_plate()
print(vehicle_type)

classifier.set_image(plate_image2)
license_number = classifier.get_license_plate()
print(license_number)
vehicle_type = classifier.parse_license_plate()
print(vehicle_type)

classifier.set_image(plate_image3)
license_number = classifier.get_license_plate()
print(license_number)
vehicle_type = classifier.parse_license_plate()
print(vehicle_type)