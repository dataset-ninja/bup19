import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import file_exists, get_file_name, get_file_name_with_ext
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    train_data_path = "/home/alex/DATASETS/TODO/BUP/BUP19/Bonn2019_P/train"
    val_data_path = "/home/alex/DATASETS/TODO/BUP/BUP19/Bonn2019_P/val"
    test_data_path = "/home/alex/DATASETS/TODO/BUP/BUP19/Bonn2019_P/eval"
    raw_splits_path = "/home/alex/DATASETS/TODO/BUP/BUP19/Bonn2019_P/train/raw_splits_rgb"

    group_tag_name = "im id"
    batch_size = 30
    rgb_folder = "/rgb/"
    depth_folder = "/depth/"
    masks_folder = "/instance/"
    images_ext = ".png"
    depth_ext = ".tiff"

    ds_name_to_split = {"train": train_data_path, "val": val_data_path, "test": test_data_path}

    def create_ann(image_path):
        labels = []
        tags = []

        group_tag = sly.Tag(group_tag_meta, get_file_name(image_path).split("_")[-1])
        tags.append(group_tag)

        if ds_name == "train":
            if get_file_name_with_ext(image_path) in alireza:
                raw = sly.Tag(alireza_meta)
            else:
                raw = sly.Tag(claus_meta)

            tags.append(raw)

        img_height = 1280
        img_wight = 720

        for mask_name in ["black.png", "green.png", "red.png", "mixed.png"]:
            mask_path = os.path.join(masks_path, get_file_name(image_path), mask_name)
            obj_class = name_to_class.get(mask_name)
            if file_exists(mask_path):
                mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
                unique_pixels = np.unique(mask_np)[1:]

                for pixel in unique_pixels:
                    mask = mask_np == pixel
                    curr_bitmap = sly.Bitmap(mask)
                    curr_label = sly.Label(curr_bitmap, obj_class)
                    labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    alireza_meta = sly.TagMeta("alireza", sly.TagValueType.NONE)
    claus_meta = sly.TagMeta("claus", sly.TagValueType.NONE)
    group_tag_meta = sly.TagMeta(group_tag_name, sly.TagValueType.ANY_STRING)

    red = sly.ObjClass("red pepper", sly.AnyGeometry, color=(255, 0, 0))
    black = sly.ObjClass("black pepper", sly.AnyGeometry, color=(0, 0, 0))
    green = sly.ObjClass("green pepper", sly.AnyGeometry, color=(0, 128, 0))
    mixed = sly.ObjClass("mixed pepper", sly.AnyGeometry, color=(128, 0, 128))

    name_to_class = {"black.png": black, "green.png": green, "red.png": red, "mixed.png": mixed}

    meta = sly.ProjectMeta(
        tag_metas=[claus_meta, alireza_meta, group_tag_meta], obj_classes=[red, black, green, mixed]
    )

    api.project.update_meta(project.id, meta.to_json())
    api.project.images_grouping(id=project.id, enable=True, tag_name=group_tag_name)

    alireza = os.listdir(os.path.join(raw_splits_path, "00_Alireza"))
    claus = os.listdir(os.path.join(raw_splits_path, "00_Claus"))

    for ds_name, data_path in ds_name_to_split.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        rgb_images_path = os.path.join(data_path, "rgb")
        depth_images_path = os.path.join(data_path, "depth")
        masks_path = os.path.join(data_path, "instance")

        images_names = os.listdir(rgb_images_path)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for imgs_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = []
            images_names_batch = []
            for im_name in imgs_names_batch:
                images_names_batch.append(im_name)
                images_pathes_batch.append(os.path.join(rgb_images_path, im_name))

                depth_name = im_name.replace(images_ext, depth_ext)
                images_names_batch.append(depth_name)
                images_pathes_batch.append(os.path.join(depth_images_path, depth_name))

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = []
            for i in range(0, len(images_pathes_batch), 2):
                ann = create_ann(images_pathes_batch[i])
                anns.extend([ann, ann])
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))

    return project
