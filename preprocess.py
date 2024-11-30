from PIL import Image, ImageOps, ImageFilter
import numpy as np
import scipy.ndimage as ndimage

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
    i = image.width-1
    start = i
    slice_count = 0
    flag = True

    while flag and i >= 0:
        value = row_histogram[i]
        if value > 0 and not in_slice:
            in_slice = True
            start = i
        elif value == 0 and in_slice:
            if slice_count < 7:
                in_slice = False
                cropped_image = image.crop((i, 0, start, image.height))
                slices.append(cropped_image)
                slice_count += 1
            else:
                in_slice = False
                cropped_image = image.crop((i, 0, start, image.height))
                column_histogram = _get_column_histogram(cropped_image)
                _, merged_flag = _slice_horizontally(cropped_image, column_histogram, return_flag=True)
                if not merged_flag:
                    slices.append(cropped_image)
                    slice_count += 1
                else:
                    flag = False
        i -= 1

    slices.reverse()

    return slices

def _slice_horizontally(image, column_histogram, return_flag=False):
    slices = []
    in_slice = False
    start = 0
    merged_flag = False

    for i, value in enumerate(column_histogram):
        if value > 0 and not in_slice:
            in_slice = True
            start = i
        elif value == 0 and in_slice:
            in_slice = False
            cropped_image = image.crop((0, start, image.width, i))
            slices.append(cropped_image)
    
    merged_width = 0
    merged_height = 0

    if len(slices) != 1:
        merged_flag = True

    for slice in slices:
        width, height = slice.size
        merged_width = max(merged_width, width)
        merged_height += height
    
    merged = Image.new("L", (merged_width, merged_height+(int(merged_height/50))), 255)
    offset = 0

    for slice in slices:
        merged.paste(slice, (0, offset))
        offset += (slice.height + int(merged_height/50))

    if return_flag:
        return merged, merged_flag
    else:
        return merged
        
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
    filter_size = image.height // 30
    if (filter_size) % 2 == 0:
        filter_size += 1

    denoised = image.filter(ImageFilter.MedianFilter(size=filter_size))
    denoised = ImageOps.invert(denoised)
    object_threshold = denoised.height // 3

    denoised = np.array(denoised)
    labeled, num_features = ndimage.label(denoised)
    object_slices = ndimage.find_objects(labeled)

    mask = np.zeros_like(denoised, dtype=np.uint8)

    for i, object_slice in enumerate(object_slices, start=1):
        if object_slice is not None:
            y_slice, x_slice = object_slice
            width = x_slice.stop - x_slice.start
            height = y_slice.stop - y_slice.start

            if width < object_threshold and height < object_threshold:
                mask[labeled==i] = 255

    denoised[mask==255] = 0
    denoised = Image.fromarray(denoised)
    denoised = ImageOps.invert(denoised)

    return denoised

def preprocess(image, threshold=128, size=28, invert=True):
    original = image
    image = _denoise(image)
    #image = image.point(lambda p: p> threshold and 255)
    slices = _slice_image(image, threshold)
    return_flag = False
    
    while threshold < 256 and threshold >= 0 and (len(slices)!=7 and len(slices)!=8 and len(slices)!=9):
        if threshold == 0:
            threshold = 128
            return_flag = True

        if return_flag:
            threshold += 1
        else:
            threshold -= 1

        image = original.point(lambda p: p> threshold and 255)
        slices = _slice_image(image, threshold)
    
    if threshold==256:
        raise ValueError("There is no numbers detected")

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