import os
import sys
import numpy as np
import random
from tests import empty_directory
from trdg.run import parse_arguments
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
from trdg.data_generator import FakeTextDataGenerator
from trdg.string_generator import (
    create_strings_from_dict,
    create_strings_from_file,
    create_strings_from_wikipedia,
    create_strings_randomly,
)
from trdg.utils import load_dict, load_fonts

sys.path.append(os.path.join(os.path.dirname(__file__), "trdg"))


special_punctuation_label_replace_dict = {'（': '(', '）': ')', 'ɡ': 'g', '“': '"', '”': '"', '〔': '[', '〕': ']', '【': '[', '】': ']', 'UNKNOWN': '·', '—': '一', '～':'~', '！': '!', '×':'x', '●': '·', '∶': ':', '︰': ':', '＝': '=', '⑴': '(1)', '⑵': '(2)', '⑶': '(3)', '⑷': '(4)', '－': '一', '―': '一', '＜': '<', '＿': '_', '／': '/', '┅': '···', 'Ｃ': 'C', '─': '-', 'Ｏ': 'O', 'Ｈ': 'H', '１': '1', '２': '2', '３': '3', '４': '4', 'Ｎ': 'N'}


def RGB_to_Hex(rgb):
    """RGB格式颜色转换为16进制颜色格式"""
    color = '#'
    for i in rgb:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color


def get_char_dict(char_dict_path):
    with open(char_dict_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    chars_dict = {}
    chars = ''
    for i in range(len(lines)):
        line = lines[i]
        char = line.replace('\n', '').replace('\t', '').replace(' ', '')
        chars_dict[char] = i
        chars += char

    char_set = list(set(chars))
    assert len(char_set) == len(chars), f'len(char_set): {len(char_set)}, len(chars): {len(chars)}'
    return chars, chars_dict


def gen_imgs(out_dir: str, name_format=3, count=5, words_len_range=[1, 25], img_height=48, distort=True, fonts_dir='./trdg/fonts/multi_fonts/ch', save=True, special_char_replace_dict=special_punctuation_label_replace_dict, debug=True):
    """_summary_

    Args:
        out_dir (str): 
        name_format (_type_): 3: write img into outpit/images and label.txt into output/labels and labels.txt.
        count (int, optional): _description_. Defaults to 5.
        language (str, optional): _description_. Defaults to 'cn'.
        words_len_range (list, optional): _description_. Defaults to [1, 13].
        img_height (int, optional): _description_. Defaults to 48.
        distort (bool, optional): _description_. Defaults to True.
        fonts_dir (str, optional): _description_. Defaults to 'F:/nets/OCR/ocr_optimize/paddleocr/PaddleOCR/doc/multi_fonts/ch'.
        save (bool, optional): _description_. Defaults to True.
        debug (bool, optional): _description_. Defaults to True.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # Argument parsing
    args = parse_arguments()

    args.output_dir = out_dir
    args.name_format = name_format
    tmp = random.random()
    if tmp < 0.2:
        args.language = 'en'
        args.fonts_dir = './trdg/fonts/latin'
    else:
        args.language = 'cn'
        args.fonts_dir = './trdg/fonts/cn'

    args.length = random.randint(words_len_range[0], words_len_range[1])

    args.count = count

    args.format = img_height

    # distort randomly
    if distort:
        if random.random() < 0.2:
            tmp = random.random()
            if tmp < 0.25:
                args.distorsion = 0
            elif tmp < 0.5:
                args.distorsion = 1
            elif tmp < 0.75:
                args.distorsion = 2
            else:
                args.distorsion = 3

            tmp = random.random()
            if tmp < 0.2:
                args.distorsion_orientation = 0
            elif tmp < 0.4:
                args.distorsion_orientation = 1
            else:
                args.distorsion_orientation = 2


    # set random color below (30,30,30)
    text_color_rgb = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
    color = RGB_to_Hex(text_color_rgb)
    args.text_color = color

    # set stroke fill
    args.stroke_fill = color

    # set gray
    args.image_mode = 'L'

    # # set inverse color
    # if random.random() < 0.00:
    #     args.color_inverse = True

    # set blur randomly, radius can only be 1
    if random.random() < 0.02:
        args.blur = 1

    # set background
    args.background = 1

    # set the space width
    tmp = random.random()
    if tmp < 0.2:
        args.space_width = 1e-4
    elif tmp < 0.4:
        args.space_width = random.random()
    elif tmp < 0.9:
        args.space_width = random.randint(1, 2)
    else:
        args.space_width = random.randint(1, 5)

    # set alignment
    tmp = random.random()
    if tmp < 0.33:
        args.alignment = 0
    elif tmp < 0.66:
        args.alignment = 1
    else:
        args.alignment = 2

    # set margins around text when rendered
    args.margins = (random.randint(0, 5), random.randint(0, 5), random.randint(0, 5), random.randint(0, 5))

    # random to generate char with 90 rotation and 270 rotation
    if random.random() < 0.1:
        args.length = 1
        tmp = random.random()
        if tmp < 0.4:
            args.skew_angle = 90
        elif tmp < 0.8:
            args.skew_angle = 270
        else:
            args.skew_angel = 90
            args.random_skew = True

    # set font and stroke width
    fonts_fs = os.listdir(args.fonts_dir)
    choice_idx = random.randint(0, len(fonts_fs)-1)
    font_name = fonts_fs[choice_idx]
    args.font = os.path.join(args.fonts_dir, font_name)
    args.stroke_width = random.randint(0, 1)

    # Create the directory if it does not exist.
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Creating word list
    if args.dict:
        lang_dict = []
        if os.path.isfile(args.dict):
            with open(args.dict, "r", encoding="utf8", errors="ignore") as d:
                lang_dict = [l for l in d.read().splitlines() if len(l) > 0]
        else:
            sys.exit("Cannot open dict")
    else:
        lang_dict = load_dict(
            os.path.join(os.path.dirname(__file__), "trdg", "dicts", args.language + ".txt")
        )

    # Create font (path) list
    if args.font_dir:
        fonts = [
            os.path.join(args.font_dir, p)
            for p in os.listdir(args.font_dir)
            if os.path.splitext(p)[1] == ".ttf"
        ]
    elif args.font:
        if os.path.isfile(args.font):
            fonts = [args.font]
        else:
            sys.exit("Cannot open font")
    else:
        fonts = load_fonts(args.language)

    # Creating synthetic sentences (or word)
    if args.language == 'cn':
        tmp = random.random()
        if tmp < 0.3:
            pass
        else:
            prepare_txt_dir = os.path.join(os.getcwd(), "trdg", "texts")
            if not os.path.exists(prepare_txt_dir):
                raise ValueError(f"fail to find {prepare_txt_dir}")
            fs = os.listdir(prepare_txt_dir)
            choice_fp = os.path.join(prepare_txt_dir, fs[random.randint(0, len(fs)-1)])
            args.input_file = choice_fp
    else:
        args.input_file = ''

    strings = []

    if args.input_file != "":
        strings = create_strings_from_file(args.input_file, args.count, max_length=args.length)
    elif args.random_sequences and tmp < 0.2:
        strings = create_strings_randomly(
                    args.length,
                    args.random,
                    args.count,
                    args.include_letters,
                    args.include_numbers,
                    args.include_symbols,
                    args.language,
                )
    else:
        strings = create_strings_from_dict(
            args.length, args.random, args.count, lang_dict
        )

    new_strings = []
    for j in range(len(strings)):
        string = strings[j]
        string = string.replace("\t", "").replace("\\t", "").replace('\n', '').replace('\r', '')
        if len(string) > 0:
            new_strings.append(string)
    strings = new_strings

    while len(strings) == 0:
        strings = create_strings_from_dict(
            args.length, args.random, args.count, lang_dict
        )

    new_strings = []
    for j in range(len(strings)):
        string = strings[j]
        string = string.replace("\t", "").replace("\\t", "").replace('\n', '').replace('\r', '').replace(' ', '')
        if len(string) > args.length:
            tmp_st = random.randint(0, len(string)-args.length)
            tmp_ed = tmp_st + random.randint(1, args.length)
            string = string[tmp_st: tmp_ed]
        if len(string) > 0 and len(string) <= words_len_range[1]:
            new_strings.append(string)
    strings = new_strings

    if args.case == "upper":
        strings = [x.upper() for x in strings]
    if args.case == "lower":
        strings = [x.lower() for x in strings]

    string_count = len(strings)

    dicts = []
    p = Pool(args.thread_count)
    for tmp_dict in p.imap_unordered(
            FakeTextDataGenerator.generate_from_tuple,
            zip(
                [i for i in range(0, string_count)],
                strings,
                [fonts[random.randrange(0, len(fonts))] for _ in range(0, string_count)],
                [args.output_dir] * string_count,
                [args.format] * string_count,
                [args.extension] * string_count,
                [args.skew_angle] * string_count,
                [args.random_skew] * string_count,
                [args.blur] * string_count,
                [args.random_blur] * string_count,
                [args.background] * string_count,
                [args.distorsion] * string_count,
                [args.distorsion_orientation] * string_count,
                [args.handwritten] * string_count,
                [args.name_format] * string_count,
                [args.width] * string_count,
                [args.alignment] * string_count,
                [args.text_color] * string_count,
                [args.orientation] * string_count,
                [args.space_width] * string_count,
                [args.character_spacing] * string_count,
                [args.margins] * string_count,
                [args.fit] * string_count,
                [args.output_mask] * string_count,
                [args.word_split] * string_count,
                [args.image_dir] * string_count,
                [args.stroke_width] * string_count,
                [args.stroke_fill] * string_count,
                [args.image_mode] * string_count,
                [args.output_bboxes] * string_count,
                [args.color_inverse] * string_count,
                [save] * string_count,
            ),
        ):
        if tmp_dict is not None:
            dicts.append(tmp_dict)
        else:
            pass

    p.terminate()

    if args.name_format == 2:
        # Create file with filename-to-label connections
        with open(
            os.path.join(args.output_dir, "labels.txt"), "a+", encoding="utf8"
        ) as f:
            for i in range(string_count):
                file_name = str(i) + "." + args.extension
                label = strings[i]
                if args.space_width == 0:
                    label = label.replace(" ", "")
                f.write("{} {}\n".format(file_name, label))
    elif args.name_format == 3:
        with open(
            os.path.join(args.output_dir, "labels.txt"), "a+", encoding="utf8"
        ) as f:
            for i in range(len(dicts)):
                tmp_dict = dicts[i]
                file_name = os.path.join(tmp_dict['image_name'])
                label = strings[i]
                if args.space_width == 0:
                    label = label.replace(" ", "")
                else:
                    label = label
                new_label = ''
                for tmp in label:
                    if tmp in special_char_replace_dict.keys():
                        new_label += special_char_replace_dict[tmp]
                    else:
                        new_label += tmp

                f.write("{}\t{}\n".format(file_name, label))
        return dicts


if __name__ == "__main__":
    out_dir = "F:/nets/OCR/ocr_optimize/data/synth1/train"
    dicts = []
    for i in range(2000):
        print(f"batch index: {i}")
        tmp_dict = gen_imgs(out_dir=out_dir, count=5, save=True)
        dicts.append(tmp_dict)
    print(dicts)
