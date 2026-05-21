# -*- coding: utf-8 -*-
import hashlib
import os
from fontTools.ttLib import TTFont
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import collections

def traverse_font_files(directory):
    """
    Traverse all font files in the given directory.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.ttf', '.otf')):
                yield os.path.join(root, file)


def get_cmap(font_file):
    """
    Get unicode cmap - Character to Glyph Index Mapping Table

    font_file: path of font file
    """
    try:
        font = TTFont(font_file)
    except:
        return None

    try:
        cmap = font.getBestCmap()
    except:
        return None
    font.close()
    return cmap


def get_decimal_unicode(font_file):
    """
    Get unicode (decimal mode - radix=10) of font.
    """
    cmap = get_cmap(font_file)
    if cmap is None:
        return None
    try:
        decimal_unicode = list(cmap.keys())
    except:
        decimal_unicode = None
    return decimal_unicode


def decimal_to_hex(decimal_unicode, prefix='uni'):
    """
    Convert decimal unicode (radix=10) to hex unicode (radix=16, str type)
    """

    def _regularize(single_decimal_unicode, prefix):
        # result of hex() contains prefix '0x', such as '0x61',
        # while font file usually use 'uni0061',
        # so support changing prefix and filling to width 4 with 0
        h = hex(single_decimal_unicode)
        single_hex_unicode = prefix + h[2:].zfill(4)
        return single_hex_unicode

    is_single_code = False
    if not isinstance(decimal_unicode, (list, tuple)):
        decimal_unicode = [decimal_unicode]
        is_single_code = True

    hex_unicode = [_regularize(x, prefix) for x in decimal_unicode]

    if is_single_code:
        hex_unicode = hex_unicode[0]
    return hex_unicode


def decimal_to_char(decimal_unicode):
    """
    Convert decimal unicode (radix=10) to characters
    """
    is_single_code = False
    if not isinstance(decimal_unicode, (list, tuple)):
        decimal_unicode = [decimal_unicode]
        is_single_code = True

    char = [chr(x) for x in decimal_unicode]

    if is_single_code:
        char = char[0]
    return char


def get_bbox_offset(bbox, image_size):
    """
    Get offset (x, y) for moving bbox to the center of image

    bbox: bounding box of character, containing [xmin, ymin, xmax, ymax]
    """
    if not isinstance(image_size, (list, tuple)):
        image_size = (image_size, image_size)

    center_x = image_size[0] // 2
    center_y = image_size[1] // 2
    xmin, ymin, xmax, ymax = bbox
    bbox_xmid = (xmin + xmax) // 2
    bbox_ymid = (ymin + ymax) // 2
    offset_x = center_x - bbox_xmid
    offset_y = center_y - bbox_ymid
    return offset_x, offset_y


def char_to_image(char, font_pil, image_size, bg_color=255, fg_color=0):
    """
    Generate an image containing single character in a font.

    char: such as '中' , 'a' ...
    font_pil: result of PIL.ImageFont
    """
    try:
        bbox = font_pil.getbbox(char)
    except Exception as e:
        return None

    if bbox is None or all(val == 0 for val in bbox):
        return None

    if not isinstance(image_size, (list, tuple)):
        image_size = (image_size, image_size)
    offset_x, offset_y = get_bbox_offset(bbox, image_size)
    offset = (offset_x, offset_y)

    # convert ttf/otf to bitmap image using PIL
    image = Image.new('L', image_size, bg_color)
    draw = ImageDraw.Draw(image)
    draw.text(offset, char, font=font_pil, fill=fg_color)

    #char_hash = hash(image.tobytes())
    #return image, char_hash

    pixels = image.load()
    for y in range(image_size[1]):
        for x in range(image_size[0]):
            if pixels[x, y] != bg_color:  # If any pixel is not background color, the character is rendered
                return image
    return None


def is_valid_unicode(code, valid_unicode_set):
    return code in valid_unicode_set

def calculate_hashes(font_pil, charset, canvas_size, x_offset, y_offset, bg_color=255):
    hash_count = collections.defaultdict(int)
    all_images = {}

    for char in charset:
        image = char_to_image(char, font_pil, canvas_size, bg_color)
        if image is not None:
            image_bytes = image.tobytes()
            hash_value = hashlib.md5(image_bytes).hexdigest()
            hash_count[hash_value] += 1
            all_images[char] = image

    # 从 hash_count 字典中提取出那些出现次数大于1的哈希值
    recurring_hashes = [hash for hash, count in hash_count.items() if count > 1]
    filtered_unicode = [char for char in all_images if hashlib.md5(all_images[char].tobytes()).hexdigest() in recurring_hashes]
    return recurring_hashes, filtered_unicode, all_images

def get_characters_from_images(directory):
    """
    Get characters from image filenames in the given directory.
    """
    characters = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract character from filename (assuming filename is the character itself)
                char = os.path.splitext(file)[0]
                characters.add(char)
    print(f"Extracted characters: {characters}")
    return list(characters)

def font2image(font_file,
               font_size,
               image_width,
               image_height,
               out_folder=None,
               valid_unicode_set=None,
               name_mode='char',
               image_extension='png',
               bg_color=255,
               fg_color=0,
               is_skip=True):
    """
    Generate images from a font.

    font_size: size of font when reading by PIL, type=float
    image_size: image_size should normally be larger than font_size
    decimal_unicode: if not None, only generate images of decimal_unicode
    name_mode: if not 'char', then will be like 'uni0061'
    is_skip: whether skip existed images
    """
    if out_folder is None:
        out_folder = os.path.splitext(font_file)[0]

    if os.path.exists(out_folder) and is_skip:
        print(f"Output folder '{out_folder}' already exists. Skipping font: {font_file}")
        return

    font_pil = ImageFont.truetype(font_file, font_size)

    decimal_unicode = get_decimal_unicode(font_file)

    if decimal_unicode is None:
        print(f"No unicode values found for font: {font_file}")
        return

    os.makedirs(out_folder, exist_ok=True)

    recurring_hashes, filtered_unicode, all_images = calculate_hashes(font_pil, [chr(code) for code in decimal_unicode],
                                                                      (image_width, image_height), 0, 0, bg_color)

    if valid_unicode_set is not None:
        decimal_unicode = [code for code in decimal_unicode if
                           is_valid_unicode(code, valid_unicode_set)]

    for code in decimal_unicode:
        char = chr(code)
        # get output filename
        #if name_mode == 'char':
        #    filename = char
        #else:
        #    filename = decimal_to_hex(code)
        filename = os.path.join(out_folder, f'{char}.{image_extension}')

        # skip existed images
        if is_skip and os.path.exists(filename):
            continue

        if char in filtered_unicode:
            #print(f"Character '{char}' is filtered out due to recurring hash, skipping.")
            continue

        image = all_images.get(char)

        if image is None:
            continue

        try:
            image.save(filename)
        except:
            pass


if __name__ == '__main__':
    font_directory = 'fontdata_example/ttf'
    text_image_directory = 'fontdata_example/font/train/chinese/Qing niao Hua guang Yao ti Font-Simplified Chinese'
    valid_unicode_set = get_characters_from_images(text_image_directory)
    valid_unicode_set = {ord(char) for char in valid_unicode_set}
    image_width = 64
    image_height = 64
    for font_file in traverse_font_files(font_directory):
        print(font_file)
        font2image(font_file, 60, image_width, image_height, valid_unicode_set=valid_unicode_set)
