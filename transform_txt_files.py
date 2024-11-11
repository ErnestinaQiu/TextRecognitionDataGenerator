import os
from trdg.utils import transform_txt_to_utf8

if __name__ == "__main__":
    source_dir = os.path.join(os.getcwd(), "trdg", "texts")
    target_dir = os.path.join(os.getcwd(), "trdg", "texts_utf8")
    transform_txt_to_utf8(source_dir=source_dir, target_dir=target_dir)
