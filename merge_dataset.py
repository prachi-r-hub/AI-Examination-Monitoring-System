import os
import shutil

# ==========================
# PROJECT PATHS
# ==========================

BASE_DIR = "Classroom_Dataset"

OUTPUT_DIR = os.path.join(BASE_DIR, "Merged_Dataset")

# ==========================
# FINAL CLASS IDS
# ==========================
#
# 0 -> Notebook (Book)
# 1 -> Paper
# 2 -> Pen
# 3 -> Pencil
# 4 -> Phone
# 5 -> Smartwatch
# 6 -> Earbuds
#

DATASETS = {

    "Notebook": {
        "folder": "Notebook",
        "old_class": 0,
        "new_class": 0
    },

    "Paper": {
        "folder": "Paper",
        "old_class": 0,
        "new_class": 1
    },

    "Pen": {
        "folder": "Pen",
        "old_class": 0,
        "new_class": 2
    },

    "Pencil": {
        "folder": "Pencil",
        "old_class": 0,
        "new_class": 3
    },

    "Phone": {
        "folder": "Phone",
        "old_class": 1,
        "new_class": 4
    },

    "Smartwatch": {
        "folder": "Smartwatch",
        "old_class": 0,
        "new_class": 5
    },

    "Earbuds": {
        "folder": "Earbuds",
        "old_class": 0,
        "new_class": 6
    }

}


# ==========================
# CREATE OUTPUT FOLDERS
# ==========================

for split in ["train", "valid", "test"]:

    os.makedirs(
        os.path.join(OUTPUT_DIR, split, "images"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(OUTPUT_DIR, split, "labels"),
        exist_ok=True
    )

print("Folders created successfully.")

# ==========================
# MERGE DATASETS
# ==========================

image_counter = 1

for dataset_name, info in DATASETS.items():

    dataset_path = os.path.join(BASE_DIR, info["folder"])

    print(f"\nProcessing {dataset_name}...")

    for split in ["train", "valid", "test"]:

        image_folder = os.path.join(dataset_path, split, "images")
        label_folder = os.path.join(dataset_path, split, "labels")

        output_image_folder = os.path.join(OUTPUT_DIR, split, "images")
        output_label_folder = os.path.join(OUTPUT_DIR, split, "labels")

        if not os.path.exists(image_folder):
            continue

        for image_file in os.listdir(image_folder):

            if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_name = f"{dataset_name.lower()}_{image_counter}"

            image_counter += 1

            extension = os.path.splitext(image_file)[1]

            new_image_name = image_name + extension
            new_label_name = image_name + ".txt"

            old_image_path = os.path.join(image_folder, image_file)
            new_image_path = os.path.join(output_image_folder, new_image_name)

            shutil.copy(old_image_path, new_image_path)

            old_label_path = os.path.join(
                label_folder,
                os.path.splitext(image_file)[0] + ".txt"
            )

            new_label_path = os.path.join(
                output_label_folder,
                new_label_name
            )

            if not os.path.exists(old_label_path):
                continue

            new_lines = []

            with open(old_label_path, "r") as f:

                for line in f:

                    line = line.strip()

                    if line == "":
                        continue

                    parts = line.split()

                    old_class = int(parts[0])

                    # Ignore unwanted classes
                    if old_class != info["old_class"]:
                        continue

                    parts[0] = str(info["new_class"])

                    new_lines.append(" ".join(parts))

            with open(new_label_path, "w") as f:

                for line in new_lines:
                    f.write(line + "\n")

print("\nAll datasets merged successfully!")


# ==========================
# CREATE data.yaml
# ==========================

yaml_text = """train: train/images
val: valid/images
test: test/images

nc: 7

names:
  0: Notebook
  1: Paper
  2: Pen
  3: Pencil
  4: Phone
  5: Smartwatch
  6: Earbuds
"""

yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")

with open(yaml_path, "w") as f:
    f.write(yaml_text)

print("\ndata.yaml created successfully!")

# ==========================
# VERIFY DATASET
# ==========================

print("\n========== DATASET SUMMARY ==========")

for split in ["train", "valid", "test"]:

    image_folder = os.path.join(OUTPUT_DIR, split, "images")
    label_folder = os.path.join(OUTPUT_DIR, split, "labels")

    image_count = len([
        file for file in os.listdir(image_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    label_count = len([
        file for file in os.listdir(label_folder)
        if file.endswith(".txt")
    ])

    print(f"{split.upper()}")

    print(f"Images : {image_count}")

    print(f"Labels : {label_count}")

    print("------------------------------")

print("Merged dataset created successfully!")
print("Location:", OUTPUT_DIR)
print("\nNow train using:\n")
print("yolo detect train model=yolov8s.pt data=Classroom_Dataset/Merged_Dataset/data.yaml epochs=100 imgsz=640")