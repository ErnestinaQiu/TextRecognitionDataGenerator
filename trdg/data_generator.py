import os
import json
import time
import numpy as np
import random
import random as rnd
import logging
from logging import getLogger

from PIL import Image, ImageFilter, ImageStat

from trdg import computer_text_generator, background_generator, distorsion_generator
from trdg.utils import mask_to_bboxes, make_filename_valid

try:
    from trdg import handwritten_text_generator
except ImportError as e:
    print("Missing modules for handwritten text generation.")

logger = getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


class FakeTextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
        Same as generate, but takes all parameters as one tuple
        return (dict): {'index': int, 'image_name': str, 'image': PIL.Image.Image, "text": text} or 
                        {"index": index, "image": final_image, "mask_image": final_mask, "image_name": image_name, "mask_name": mask_name, "text": text}
        """
        ret = cls.generate(*t)
        return ret

    @classmethod
    def generate(
        cls,
        index: int,
        text: str,
        font: str,
        out_dir: str,
        size: int,
        extension: str,
        skewing_angle: int,
        random_skew: bool,
        blur: int,
        random_blur: bool,
        background_type: int,
        distorsion_type: int,
        distorsion_orientation: int,
        is_handwritten: bool,
        name_format: int,
        width: int,
        alignment: int,
        text_color: str,
        orientation: int,
        space_width: int,
        character_spacing: int,
        margins: int,
        fit: bool,
        output_mask: bool,
        word_split: bool,
        image_dir: str,
        stroke_width: int = 0,
        stroke_fill: str = "#282828",
        image_mode: str = "RGB",
        output_bboxes: int = 0,
        color_inverse: bool = False,
        save: bool = False,
    ) -> dict:
        if name_format == 4:
            save = False

        image = None

        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        ##########################
        # Create picture of text #
        ##########################
        if is_handwritten:
            if orientation == 1:
                raise ValueError("Vertical handwritten text is unavailable")
            image, mask = handwritten_text_generator.generate(text, text_color)
        else:
            image, mask = computer_text_generator.generate(
                text,
                font,
                text_color,
                size,
                orientation,
                space_width,
                character_spacing,
                fit,
                word_split,
                stroke_width,
                stroke_fill,
            )
        random_angle = rnd.randint(0 - skewing_angle, skewing_angle)

        rotated_img = image.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        rotated_mask = mask.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        #############################
        # Apply distortion to image #
        #############################
        if distorsion_type == 0:
            distorted_img = rotated_img  # Mind = blown
            distorted_mask = rotated_mask
        elif distorsion_type == 1:
            distorted_img, distorted_mask = distorsion_generator.sin(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        elif distorsion_type == 2:
            distorted_img, distorted_mask = distorsion_generator.cos(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        else:
            distorted_img, distorted_mask = distorsion_generator.random(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )

        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        if orientation == 0:
            try:
                new_width = int(
                    distorted_img.size[0]
                    * (float(size - vertical_margin) / float(distorted_img.size[1]+1e-8))
                )
                resized_img = distorted_img.resize(
                    (new_width, size - vertical_margin), Image.Resampling.LANCZOS
                )
                resized_mask = distorted_mask.resize(
                    (new_width, size - vertical_margin), Image.Resampling.NEAREST
                )
                background_width = width if width > 0 else new_width + horizontal_margin
                background_height = size
            except Exception as e:
                logger.error(f"new_width: {new_width}, size: {size}, vertical_margin: {vertical_margin}, new_width: {new_width}" + str(e))
                return

        # Vertical text
        elif orientation == 1:
            new_height = int(
                float(distorted_img.size[1])
                * (float(size - horizontal_margin) / float(distorted_img.size[0]))
            )
            resized_img = distorted_img.resize(
                (size - horizontal_margin, new_height), Image.Resampling.LANCZOS
            )
            resized_mask = distorted_mask.resize(
                (size - horizontal_margin, new_height), Image.Resampling.NEAREST
            )
            background_width = size
            background_height = new_height + vertical_margin
        else:
            raise ValueError("Invalid orientation")

        #############################
        # Generate background image #
        #############################
        if background_type == 0:
            background_img = background_generator.gaussian_noise(
                background_height, background_width
            )
        elif background_type == 1:
            background_img = background_generator.plain_white(
                background_height, background_width
            )
        elif background_type == 2:
            background_img = background_generator.quasicrystal(
                background_height, background_width
            )
        else:
            background_img = background_generator.image(
                background_height, background_width, image_dir
            )
        background_mask = Image.new(
            "RGB", (background_width, background_height), (0, 0, 0)
        )

        ##############################################################
        # Comparing average pixel value of text and background image #
        ##############################################################
        try:
            resized_img_st = ImageStat.Stat(resized_img, resized_mask.split()[2])
            background_img_st = ImageStat.Stat(background_img)

            resized_img_px_mean = sum(resized_img_st.mean[:2]) / 3
            background_img_px_mean = sum(background_img_st.mean) / 3

            if abs(resized_img_px_mean - background_img_px_mean) < 15:
                print("value of mean pixel is too similar. Ignore this image")

                print("resized_img_st \n {}".format(resized_img_st.mean))
                print("background_img_st \n {}".format(background_img_st.mean))

                return
        except Exception as err:
            return

        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = resized_img.size

        if alignment == 0 or width == -1:
            background_img.paste(resized_img, (margin_left, margin_top), resized_img)
            background_mask.paste(resized_mask, (margin_left, margin_top))
        elif alignment == 1:
            background_img.paste(
                resized_img,
                (int(background_width / 2 - new_text_width / 2), margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (int(background_width / 2 - new_text_width / 2), margin_top),
            )
        else:
            background_img.paste(
                resized_img,
                (background_width - new_text_width - margin_right, margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (background_width - new_text_width - margin_right, margin_top),
            )

        ############################################
        # Change image mode (RGB, grayscale, etc.) #
        ############################################

        background_img = background_img.convert(image_mode)
        background_mask = background_mask.convert(image_mode)

        #######################
        # Apply gaussian blur #
        #######################

        gaussian_filter = ImageFilter.GaussianBlur(
            radius=blur if not random_blur else rnd.random() * blur
        )
        final_image = background_img.filter(gaussian_filter)
        final_mask = background_mask.filter(gaussian_filter)

        #############################
        # inverse color, not finish #
        #############################
        if color_inverse:
            if image_mode in ['RGB', 'L']:
                resized_img = 255 - np.asarray(resized_img)
                if image_mode == "L":
                    resized_img = Image.fromarray(resized_img, mode='L')
                elif image_mode == "RGB":
                    resized_img = rgb2gray(rgb=resized_img)
                    if len(resized_img.shape) == 3:
                        raise ValueError(f"resized_img.shape: {resized_img.shape}")
                    resized_img = Image.fromarray(resized_img, mode='L')

            elif image_mode in ['1']:
                resized_img = 1 - np.asarray(resized_img)
                resized_img = Image.fromarray(resized_img, mode='1')

        if save:
            #####################################
            # Generate name for resulting image #
            #####################################
            # We remove spaces if space_width == 0
            if space_width == 0:
                text = text.replace(" ", "")
            if name_format == 0:
                name = "{}_{}".format(text, str(index))
            elif name_format == 1:
                name = "{}_{}".format(str(index), text)
            elif name_format == 2:
                name = str(index)
            elif name_format == 3:
                name = "{}_{}".format(text, str(index))
            else:
                print("{} is not a valid name format. Using default.".format(name_format))
                name = "{}_{}".format(text, str(index))

            name = make_filename_valid(name, allow_unicode=True)
            image_name = "{}.{}".format(name, extension)

            loop_st = time.time()
            loop_ed = time.time()
            while image_name in os.listdir(out_dir) and (loop_ed - loop_st) < 5:
                name = "_".join([name, str(random.randint(0, 9))])
                image_name = "{}.{}".format(name, extension)
                loop_ed = time.time()
            if image_name in os.listdir(out_dir):
                name = "_".join([name, str(int(time.time()))])
                image_name = "{}.{}".format(name, extension)

            mask_name = "{}_mask.png".format(name)
            box_name = "{}_boxes.txt".format(name)
            tess_box_name = "{}.box".format(name)

            # label_name = ".".join([name, "txt"])

            # Save the image
            if out_dir is not None:
                # image_dir = os.path.join(out_dir, "images")
                image_dir = out_dir
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, image_name)
                try:
                    final_image.save(image_path)
                except Exception as e:
                    print(f"image_name: {image_name}")
                    raise e
                if output_mask == 1:
                    final_mask.save(os.path.join(image_dir, mask_name))
                if output_bboxes == 1:
                    bboxes = mask_to_bboxes(final_mask)
                    with open(os.path.join(out_dir, box_name), "w") as f:
                        for bbox in bboxes:
                            f.write(" ".join([str(v) for v in bbox]) + "\n")
                if output_bboxes == 2:
                    bboxes = mask_to_bboxes(final_mask, tess=True)
                    with open(os.path.join(image_dir, tess_box_name), "w") as f:
                        for bbox, char in zip(bboxes, text):
                            f.write(
                                " ".join([char] + [str(v) for v in bbox] + ["0"]) + "\n"
                            )

                # # save the labels
                # if space_width == 0:
                #     text = text.replace(" ", "")

                # label_out_dir = os.path.join(out_dir, "labels")
                # os.makedirs(label_out_dir, exist_ok=True)
                # msg = f'{image_path}\t{text}'
                # with open(os.path.join(label_out_dir, label_name), 'w+', encoding='utf8') as f:
                #     f.write(msg)

        if name_format != 4:
            if output_mask == 1:
                return {"index": index, "image": np.asarray(final_image), "mask_image": np.asarray(final_mask), "image_name": image_name, "mask_name": mask_name, "text": text}
            return {"index": index, "image": np.asarray(final_image), "image_name": image_name, "text": text}
        else:
            return {"image": np.asarray(final_image), "text": text}

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
