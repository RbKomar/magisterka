import cv2
import os
import random
import numpy as np
import imutils

IMGS_DIR = r"imgs"


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


def random_augmentation(hair_img, prob=0.8, rotation_range=180, gamma=0.8):
    odds = 100 * prob
    nodds = 100 * (1 - prob)
    rotation_flag = bool(random.choice([] + ([True] * int(odds)) + ([False] * int(nodds))))
    darken_flag = bool(random.choice([] + ([True] * int(odds)) + ([False] * int(nodds))))
    dst = hair_img
    if rotation_flag:
        dst = imutils.rotate_bound(hair_img, random.randint(40, rotation_range))
    if darken_flag:
        dst = adjust_gamma(dst, gamma=gamma)

    return dst


def fix_size(hair, skin):
    if hair.shape[0] >= skin.shape[0]:
        dim = (hair.shape[1], skin.shape[0] - 10)
        hair = cv2.resize(hair, dim, interpolation=cv2.INTER_AREA)
    if hair.shape[1] >= skin.shape[1]:
        dim = (skin.shape[1] - 10, hair.shape[0])
        hair = cv2.resize(hair, dim, interpolation=cv2.INTER_AREA)
    return hair


def get_number_of_tiles(hair, skin):
    rows_skin_patch, cols_skin_patch, _ = skin.shape
    rows_single_hair, cols_single_hair, _ = hair.shape
    n_rows = rows_skin_patch // rows_single_hair
    n_cols = cols_skin_patch // cols_single_hair
    return n_cols, n_rows, rows_single_hair, cols_single_hair


def get_random_hair_length():
    singles_hair = os.path.join(IMGS_DIR, "single_hair")
    hair_length_dirs = os.listdir(singles_hair)
    hair_length = random.choice(hair_length_dirs)

    single_hairs_dir = os.path.join(singles_hair, hair_length)
    single_hair_dirs = os.listdir(single_hairs_dir)
    return hair_length, single_hairs_dir, single_hair_dirs


def get_reps_and_counts(hair_length):
    n_count = 4
    n_reps = 5
    if hair_length == "long":
        n_count = 3
        n_reps = 20
    elif hair_length == "medium":
        n_count = 5
        n_reps = 8
    return n_count, n_reps


def get_hair_masks(hair):
    img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    return mask, mask_inv


def add_hair(single_hair, skin_patch):
    n_cols, n_rows, rows_single_hair, cols_single_hair = get_number_of_tiles(single_hair, skin_patch)
    mask, mask_inv = get_hair_masks(single_hair)
    mult_prob = 1
    if n_rows > 20 and n_cols > 20:
        mult_prob *= n_rows // 7

    for y in range(n_rows):
        for x in range(n_cols):
            roi = skin_patch[y * rows_single_hair:rows_single_hair * (y + 1),
                  x * cols_single_hair: cols_single_hair * (x + 1)]
            y_flag = n_rows * 0.5 - n_rows * 0.2 < y < n_rows * 0.5 + n_rows * 0.2
            x_flag = n_cols * 0.5 - n_cols * 0.2 < x < n_cols * 0.5 + n_cols * 0.2
            if y_flag or x_flag:
                add_flag = bool(random.choice([True] + ([False] * 19 * mult_prob)))
            else:
                add_flag = bool(random.choice([True] + ([False] * 9 * mult_prob)))
            if add_flag:
                skin_patch_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                single_hair_fg = cv2.bitwise_and(single_hair, single_hair, mask=mask)
                roi = cv2.add(skin_patch_bg, single_hair_fg)
            skin_patch[y * rows_single_hair:rows_single_hair * (y + 1),
            x * cols_single_hair: cols_single_hair * (x + 1)] = roi
    return skin_patch


def generate_hairs(skin_patch):
    hair_length, single_hairs_dir, single_hair_dirs = get_random_hair_length()
    n_count, n_reps = get_reps_and_counts(hair_length)
    single_hairs = random.sample(single_hair_dirs, n_count)

    for single_hair in single_hairs:
        path_to_single_hair = os.path.join(single_hairs_dir, single_hair)
        single_hair = cv2.imread(path_to_single_hair)
        for _ in range(n_reps):
            single_hair = random_augmentation(single_hair)
            single_hair = fix_size(single_hair, skin_patch)
            skin_patch = add_hair(single_hair, skin_patch)
    return skin_patch


def generate_hairy_images():
    skin_dir = os.path.join(IMGS_DIR, "without-hairs")
    hairy_skin_dir = os.path.join(IMGS_DIR, "with-hairs")
    images = os.listdir(skin_dir)
    for image in images:
        path = os.path.join(skin_dir, image)
        for i in range(5):
            skin_patch = cv2.imread(path)
            hairy_skin = generate_hairs(skin_patch)
            hairy_skin_path = os.path.join(hairy_skin_dir, image.split(".")[0] + f"_hairy_{i}.jpg")
            cv2.imwrite(hairy_skin_path, hairy_skin)


def main():
    generate_hairy_images()


if __name__ == "__main__":
    main()
