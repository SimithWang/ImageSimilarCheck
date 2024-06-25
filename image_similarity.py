import os
import shutil
import cv2
import numpy as np
import yaml
from sklearn.metrics.pairwise import cosine_similarity
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("image_similarity.log"),
                        logging.StreamHandler()
                    ])


def load_config():
    # 获取当前脚本或可执行文件所在的目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config.yaml')
    if not os.path.exists(config_path):
        logging.error(f"Configuration file {config_path} not found.")
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def load_images_from_folder(folder):
    if not os.path.exists(folder):
        logging.error(f"Input folder {folder} not found.")
        raise FileNotFoundError(f"Input folder {folder} not found.")

    images = {}
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images[filename] = img
    logging.info(f"Loaded {len(images)} images from folder {folder}")
    return images


def calculate_histogram(image):
    # Convert the image to HSV color-space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculate the histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def compare_images(hist1, hist2):
    # Compute the cosine similarity between two histograms
    hist1 = hist1.reshape(1, -1)
    hist2 = hist2.reshape(1, -1)
    return cosine_similarity(hist1, hist2)[0][0]


def compare_image_pair(image_pair):
    (name1, img1), (name2, img2), threshold = image_pair
    hist1 = calculate_histogram(img1)
    hist2 = calculate_histogram(img2)
    similarity = compare_images(hist1, hist2)
    if similarity >= threshold:
        logging.info(f"Images {name1} and {name2} are similar (similarity = {similarity:.4f})")
        return name2
    return None


def find_and_remove_similar_images(images, input_folder, output_folder, threshold):
    os.makedirs(output_folder, exist_ok=True)
    image_names = list(images.keys())
    similar_pairs = set()

    logging.info(f"Comparing {len(image_names)} images for similarity with threshold {threshold}")

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(compare_image_pair, (
            (image_names[i], images[image_names[i]]), (image_names[j], images[image_names[j]]), threshold))
            for i in range(len(image_names)) for j in range(i + 1, len(image_names))
        ]

        for future in as_completed(futures):
            result = future.result()
            if result:
                similar_pairs.add(result)

    unique_images = set(image_names) - similar_pairs

    for img in unique_images:
        shutil.copy(os.path.join(input_folder, img), os.path.join(output_folder, img))
        logging.info(f"Copied {img} to {output_folder}")


def main():
    config = load_config()
    input_folder = config['input_folder']
    output_folder = config['output_folder']
    similarity_threshold = config['similarity_threshold']

    images = load_images_from_folder(input_folder)
    find_and_remove_similar_images(images, input_folder, output_folder, threshold=similarity_threshold)


if __name__ == "__main__":
    main()
