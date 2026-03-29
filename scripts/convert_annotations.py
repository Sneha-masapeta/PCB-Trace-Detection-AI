import os

IMG_WIDTH = 640
IMG_HEIGHT = 640


def convert_bbox(x1, y1, x2, y2):
    """Convert DeepPCB bbox → YOLO normalized format"""

    w = x2 - x1
    h = y2 - y1

    cx = x1 + w / 2
    cy = y1 + h / 2

    cx /= IMG_WIDTH
    cy /= IMG_HEIGHT
    w /= IMG_WIDTH
    h /= IMG_HEIGHT

    return cx, cy, w, h


def convert_annotation_file(input_file, output_file):

    with open(input_file, "r") as f:
        lines = f.readlines()

    with open(output_file, "w") as out:

        for line in lines:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            x1, y1, x2, y2, cls = map(int, parts)

            # DeepPCB class index starts from 1
            cls = cls - 1

            cx, cy, w, h = convert_bbox(x1, y1, x2, y2)

            out.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def process_dataset(pcbdata_root, output_root):

    os.makedirs(output_root, exist_ok=True)

    # iterate over group folders
    for group in os.listdir(pcbdata_root):

        group_path = os.path.join(pcbdata_root, group)

        if not os.path.isdir(group_path):
            continue

        for subfolder in os.listdir(group_path):

            # annotation folders end with "_not"
            if subfolder.endswith("_not"):

                ann_folder = os.path.join(group_path, subfolder)

                for file in os.listdir(ann_folder):

                    if file.endswith(".txt"):

                        input_path = os.path.join(ann_folder, file)
                        output_path = os.path.join(output_root, file)

                        convert_annotation_file(input_path, output_path)

                        print(f"Converted: {input_path}")


if __name__ == "__main__":

    pcbdata_root = "/content/DeepPCB/PCBData"   # Colab path
    output_root = "/content/dataset/labels"

    process_dataset(pcbdata_root, output_root)

    print("All annotations converted successfully.")