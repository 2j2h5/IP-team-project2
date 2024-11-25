from PIL import Image
from simple_conv_net import SimpleConvNet
from classifier import Classifier

#prepare SimpleConvNet
network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
network.load_params("params.pkl")

#prepare image
image_path = 'images/number-plate-clean.jpg'
plate_image = Image.open(image_path).convert('L')

#do classification
classifier = Classifier(network)
classifier.set_image(plate_image)
classifier.get_license_plate()