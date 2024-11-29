from PIL import Image, ImageOps, ImageFilter
import numpy as np

def _get_row_histogram(image, threshold=128):
    binary_image = image.point(lambda p: p> threshold and 255)
    binary_array = np.array(binary_image)

    row_histogram = np.sum(binary_array==0, axis=0)
    return row_histogram

def _get_column_histogram(image, threshold=128):
    binary_image = image.point(lambda p: p> threshold and 255)
    binary_array = np.array(binary_image)

    column_histogram = np.sum(binary_array==0, axis=1)
    return column_histogram

def _slice_vetically(image, row_histogram):
    slices = []
    in_slice = False
    start = 0

    for i, value in enumerate(row_histogram):
        if value > 0 and not in_slice:
            in_slice = True
            start = i
        elif value == 0 and in_slice:
            in_slice = False
            cropped_image = image.crop((start, 0, i, image.height))
            slices.append(cropped_image)

    return slices

def _slice_horizontally(image, column_histogram):
    in_slice = False
    start = 0

    for i, value in enumerate(column_histogram):
        if value > 0 and not in_slice:
            in_slice = True
            start = i
        elif value == 0 and in_slice:
            in_slice = False
            cropped_image = image.crop((0, start, image.width, i))
            return cropped_image
        
def _slice_image(image, threshold=128):
    row_histogram = _get_row_histogram(image, threshold)
    vertical_slices = _slice_vetically(image, row_histogram)

    horizontal_slices = []

    for slice in vertical_slices:
        column_histogram = _get_column_histogram(slice, threshold)
        horizontal_slices.append(_slice_horizontally(slice, column_histogram))

    return horizontal_slices

def _reduce_image(image, size=28):
    aspect_ratio = image.width / image.height

    if aspect_ratio > 1:
        new_width = size-6
        new_height = int((size-6) / aspect_ratio)
    else:
        new_width = int((size-6) * aspect_ratio)
        new_height = size-6

    reduced_image = image.resize((new_width, new_height))
    padded_image = Image.new("L", (size, size), 255)

    paste_x = (size - new_width) // 2
    paste_y = (size - new_height) // 2
    padded_image.paste(reduced_image, (paste_x, paste_y))

    return padded_image

def _denoise(image):
    denoised = image.filter(ImageFilter.MedianFilter(size=3))
    return denoised

def preprocess(image, threshold=128, size=28, invert=True):
    image = _denoise(image)
    image = image.point(lambda p: p> threshold and 255)
    slices = _slice_image(image, threshold)

    while threshold < 256 and len(slices)==0:
        threshold += 1
        print(threshold)
        image = image.point(lambda p: p> threshold and 255)
        slices = _slice_image(image, threshold)

    if threshold==256:
        raise ValueError("There are no numbers detected")

    result = []

    if invert:
        for slice in slices:
            reduced_slice = _reduce_image(slice, size)
            inverted_slice = ImageOps.invert(reduced_slice)
            result.append(inverted_slice)
    else:
        for slice in slices:
            reduced_slice = _reduce_image(slice, size)
            result.append(reduced_slice)

    return result

def merge_korean(slices, threshold=128, size=28, gap=4):
    vertical_slices = []
    for slice in slices:
        slice = ImageOps.invert(slice)
        row_histogram = _get_row_histogram(slice, threshold)
        vertical_slice = _slice_vetically(slice, row_histogram)
        vertical_slices.append(vertical_slice[0])

    if len(vertical_slices) == 1:
        resized = _reduce_image(vertical_slices[0], size)
        resized = ImageOps.invert(resized)

    elif len(vertical_slices) == 2:
        consonant = vertical_slices[0]
        vowel = vertical_slices[1]

        c_width, c_height = consonant.size
        v_width, v_height = vowel.size

        merged_width = c_width + v_width + gap
        merged_height = max(c_height, v_height)

        merged = Image.new("L", (merged_width, merged_height), color=255)
        merged.paste(consonant, (0, 0))
        merged.paste(vowel, (c_width + gap, 0))
        
        column_histogram = _get_column_histogram(merged, threshold)
        horizontal_slice = _slice_horizontally(merged, column_histogram)

        resized = _reduce_image(horizontal_slice, size)
        resized = ImageOps.invert(resized)
        #resized = resized.filter(ImageFilter.SHARPEN)
        #resized = resized.filter(ImageFilter.MaxFilter(size=3))

    return resized