import argparse
import errno
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import random as rnd
import sys
from multiprocessing import Pool
from trdg.data_generator import FakeTextDataGenerator
from tqdm import tqdm
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate synthetic text data for text recognition.")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, nargs="?", help="The output directory", default="./images_out/")
    
    # Input file
    parser.add_argument("-i", "--input_file", type=str, nargs="?", help="When set, this argument uses a specified text file as source for the text", default="./texts/data.txt")
    
    # Language
    parser.add_argument("-l", "--language", type=str, nargs="?", help="The language to use", default="en")
    
    # Number of images to be created
    parser.add_argument("-c", "--count", type=int, nargs="?", help="The number of images to be created.")
    
    # Random sequences options
    parser.add_argument("-rs", "--random_sequences", action="store_true", help="Use random sequences as the source text for the generation.", default=False)
    parser.add_argument("-let", "--include_letters", action="store_true", help="Define if random sequences should contain letters.", default=False)
    parser.add_argument("-num", "--include_numbers", action="store_true", help="Define if random sequences should contain numbers.", default=False)
    parser.add_argument("-sym", "--include_symbols", action="store_true", help="Define if random sequences should contain symbols.", default=False)
    
    # Text length and randomness
    parser.add_argument("-w", "--length", type=int, nargs="?", help="Define how many words should be included in each generated sample.", default=1)
    parser.add_argument("-r", "--random", action="store_true", help="Define if the produced string will have variable word count.", default=False)
    
    # Image format and size
    parser.add_argument("-f", "--format", type=int, nargs="?", help="Define the height of the produced images if horizontal, else the width", default=64)
    
    # Number of threads to use
    parser.add_argument("-t", "--thread_count", type=int, nargs="?", help="Define the number of threads to use for image generation", default=1)
    
    # File extension
    parser.add_argument("-e", "--extension", type=str, nargs="?", help="Define the extension to save the image with", default="png")
    
    # Skew options
    parser.add_argument("-k", "--skew_angle", type=int, nargs="?", help="Define skewing angle of the generated text.", default=10)
    parser.add_argument("-rk", "--random_skew", action="store_true", help="When set, the skew angle will be randomized.", default=True)
    
    # Blur options
    parser.add_argument("-bl", "--blur", type=int, nargs="?", help="Apply gaussian blur to the resulting sample.", default=3)
    parser.add_argument("-rbl", "--random_blur", action="store_true", help="When set, the blur radius will be randomized.", default=True)
    
    # Background type
    parser.add_argument("-b", "--background", type=int, nargs="?", help="Define what kind of background to use.", default=3)
    
    # Handwriting option
    parser.add_argument("-hw", "--handwritten", action="store_true", help='Define if the data will be "handwritten" by an RNN.', default=False)
    
    # File naming format
    parser.add_argument("-na", "--name_format", type=int, help="Define how the produced files will be named.", default=2)
    
    # Output options
    parser.add_argument("-om", "--output_mask", type=int, help="Define if the generator will return masks for the text", default=0)
    parser.add_argument("-obb", "--output_bboxes", type=int, help="Define if the generator will return bounding boxes for the text.", default=0)
    
    # Distortion options
    parser.add_argument("-d", "--distorsion", type=int, nargs="?", help="Define a distortion applied to the resulting image.", default=3)
    parser.add_argument("-do", "--distorsion_orientation", type=int, nargs="?", help="Define the distortion's orientation.", default=0)
    
    # Image dimensions and alignment
    parser.add_argument("-wd", "--width", type=int, nargs="?", help="Define the width of the resulting image.", default=-1)
    parser.add_argument("-al", "--alignment", type=int, nargs="?", help="Define the alignment of the text in the image.", default=1)
    
    # Orientation
    parser.add_argument("-or", "--orientation", type=int, nargs="?", help="Define the orientation of the text.", default=0)
    
    # Text color
    parser.add_argument("-tc", "--text_color", type=str, nargs="?", help="Define the text's color.", default="#282828")
    
    # Spacing options
    parser.add_argument("-sw", "--space_width", type=float, nargs="?", help="Define the width of the spaces between words.", default=1.0)
    parser.add_argument("-cs", "--character_spacing", type=int, nargs="?", help="Define the width of the spaces between characters.", default=1)
    
    # Margins
    parser.add_argument("-m", "--margins", type=str, nargs="?", help="Define the margins around the text when rendered.", default="5,5,5,5")
    
    # Fit text
    parser.add_argument("-fi", "--fit", action="store_true", help="Apply a tight crop around the rendered text", default=False)
    
    # Font options
    parser.add_argument("-ft", "--font", type=str, nargs="?", help="Define font to be used")
    parser.add_argument("-fd", "--font_dir", type=str, nargs="?", help="Define a font directory to be used", default="./fonts/vi/")
    
    # Image directory
    parser.add_argument("-id", "--image_dir", type=str, nargs="?", help="Define an image directory to use when background is set to image", default="./images/")
    
    # Case options
    parser.add_argument("-ca", "--case", type=str, nargs="?", help="Generate upper or lowercase only.")
    
    # Dictionary file
    parser.add_argument("-dt", "--dict", type=str, nargs="?", help="Define the dictionary to be used")
    
    # Word splitting
    parser.add_argument("-ws", "--word_split", action="store_true", help="Split on words instead of on characters.", default=True)
    
    # Stroke options
    parser.add_argument("-stw", "--stroke_width", type=int, nargs="?", help="Define the width of the strokes", default=0)
    parser.add_argument("-stf", "--stroke_fill", type=str, nargs="?", help="Define the color of the contour of the strokes.", default="#282828")
    
    # Image mode
    parser.add_argument("-im", "--image_mode", type=str, nargs="?", help="Define the image mode to be used.", default="RGB")
    
    return parser.parse_args()

def stream_input_file(input_file, start_idx=0, batch_size=10000):
    batch = []
    with open(input_file, 'r', encoding='utf-8') as file:
        batch = []
        for idx, line in enumerate(file):
            if idx < start_idx:
                continue
            if terminate_flag.is_set():
                break
            batch.append(line.strip())
            if len(batch) == batch_size:
                yield idx, batch
                batch = []
        if batch:
            yield idx, batch

def create_labels_file(args):
    with open(os.path.join(args.output_dir, "labels.txt"), "w", encoding="utf8") as f:
        for idx, string in enumerate(stream_input_file(args.input_file)):
            file_name = f"{idx}.{args.extension}"
            f.write(f"{file_name}\t{string}\n")

def get_last_processed_index(output_dir, extension):
    existing_images = [f for f in os.listdir(output_dir) if f.endswith(extension)]
    if not existing_images:
        return -1
    max_index = max(int(os.path.splitext(f)[0]) for f in existing_images)
    return max_index

def process_batch(args, start_idx, batch):
    fonts = [os.path.join(args.font_dir, p) for p in os.listdir(args.font_dir) if os.path.splitext(p)[1] == ".ttf"]
    for idx, string in enumerate(batch, start=start_idx):
        FakeTextDataGenerator.generate_from_tuple(
            (idx, string, fonts[rnd.randrange(0, len(fonts))], args.output_dir, 64,  # size (image height)
            args.extension, args.skew_angle, args.random_skew, args.blur, args.random_blur,
            args.background, 0, 0, 0, args.name_format, -1, 1, args.text_color, 0, 1.0, 
            args.character_spacing, (5, 5, 5, 5), False, 0, args.word_split, args.image_dir,
            0, args.text_color, args.image_mode, 0)
        )

def main():
    args = parse_arguments()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    p = Pool(args.thread_count)

    # Determine the starting index
    start_idx = get_last_processed_index(args.output_dir, args.extension) + 1

    for batch in tqdm(stream_input_file(args.input_file, start_idx=start_idx, batch_size=10000)):
        p.apply_async(process_batch, (args, start_idx, batch))
        start_idx += len(batch)

    p.close()
    p.join()

    if args.name_format == 2:
        create_labels_file(args)

if __name__ == "__main__":
    main()