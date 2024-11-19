from PIL import Image
from preprocess import *
from simple_conv_net import SimpleConvNet

network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

network.load_params("params.pkl")

image_path = 'images/number-plate-clean.jpg'
number_plate = Image.open(image_path).convert('L')

slices = preprocess(number_plate)
for slice in slices:
    slice_array = np.array(slice)
    slice_array = slice_array[np.newaxis, np.newaxis, :, :]
    slice_array = slice_array.astype(np.float32)

    scores = network.predict(slice_array)
    predicted_label = np.argmax(scores)
    print(predicted_label)
