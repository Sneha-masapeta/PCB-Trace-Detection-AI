import os
import cv2
import numpy as np
from tqdm import tqdm

DATASET_PATH = "/content/DeepPCB/PCBData"
OUTPUT_PATH = "PCBData_processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def compute_diff(template, test):

    diff = cv2.absdiff(test, template)

    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    return diff


groups = os.listdir(DATASET_PATH)

for group in tqdm(groups):

    group_path = os.path.join(DATASET_PATH, group)

    if not os.path.isdir(group_path):
        continue

    for folder in os.listdir(group_path):

        folder_path = os.path.join(group_path, folder)

        if not os.path.isdir(folder_path):
            continue

        files = os.listdir(folder_path)

        for file in files:

            if file.endswith("_test.jpg"):

                base = file.replace("_test.jpg", "")

                test_path = os.path.join(folder_path, base + "_test.jpg")
                temp_path = os.path.join(folder_path, base + "_temp.jpg")

                test = cv2.imread(test_path)

                if test is None:
                    continue

                save_folder = os.path.join(OUTPUT_PATH, group, folder)
                os.makedirs(save_folder, exist_ok=True)

                # CASE 1 : template exists → create 4 channel
                if os.path.exists(temp_path):

                    template = cv2.imread(temp_path)

                    template = cv2.resize(template, (test.shape[1], test.shape[0]))

                    diff = compute_diff(template, test)

                    b,g,r = cv2.split(test)

                    four_channel = cv2.merge((b,g,r,diff))

                    save_path = os.path.join(save_folder, base + "_4ch.png")

                    cv2.imwrite(save_path, four_channel)

                # CASE 2 : template missing → fallback to RGB
                else:

                    save_path = os.path.join(save_folder, base + "_rgb.png")

                    cv2.imwrite(save_path, test)

print("Dataset preprocessing complete")