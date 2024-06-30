import os
import cv2
from SegTracker import SegTracker
from model_args import aot_args, sam_args, segtracker_args
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc


def create_output_dir(output_dir):
    """
    创建给定路径的目录，如果它尚不存在的话。

    参数:
    output_dir (str): 待创建或检查的输出目录路径

    返回:
    None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")
    else:
        # print(f"目录 {output_dir} 已经存在")
        pass


def save_prediction(pred_mask, output_dir, file_name):
    create_output_dir(output_dir)
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode="P")
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir, file_name))


def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode="P")
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode="RGB")
    return np.array(save_mask)


def draw_mask(img, mask, alpha=0.7, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id * 3 : id * 3 + 3]
            else:
                color = [0, 0, 0]
            foreground = img * (1 - alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = mask == id

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = mask != 0
        countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
        foreground = img * (1 - alpha) + colorize_mask(mask) * alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours, :] = 0

    return img_mask.astype(img.dtype)


video_name = "cars"
# video_name = 'cell'
# video_name = '240516-02'
# video_name = '240529_1'
# video_name = "423937009-1-208" # cups
io_args = {
    "input_video": f"./assets/{video_name}.mp4",
    "output_mask_dir": f"./assets/{video_name}_masks_with_vit_matte",  # save pred masks
    "output_video": f"./assets/{video_name}_seg.mp4",  # mask+frame vizualization, mp4 or avi, else the same as input video
    "output_gif": f"./assets/{video_name}_seg.gif",  # mask visualization
}


# init vit matte
import sys
import os

matte_anything_path = "/home/fengkaige/codespace/Matte-Anything/"
os.chdir(matte_anything_path)
sys.path.append("./")

# from matte_anything import init_vitmatte


from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer


def init_vitmatte(model_type):
    """
    Initialize the vitmatte with model_type in ['vit_s', 'vit_b']
    """
    cfg = LazyConfig.load(vitmatte_config[model_type])
    vitmatte = instantiate(cfg.model)
    vitmatte.to(device)
    vitmatte.eval()
    DetectionCheckpointer(vitmatte).load(vitmatte_models[model_type])

    return vitmatte


def generate_trimap(mask, erode_kernel_size=10, dilate_kernel_size=10):
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    eroded = cv2.erode(mask, erode_kernel, iterations=5)
    dilated = cv2.dilate(mask, dilate_kernel, iterations=5)
    trimap = np.zeros_like(mask)
    trimap[dilated == 255] = 128
    trimap[eroded == 255] = 255
    return trimap


def generate_checkerboard_image(height, width, num_squares):
    num_squares_h = num_squares
    square_size_h = height // num_squares_h
    square_size_w = square_size_h
    num_squares_w = width // square_size_w

    new_height = num_squares_h * square_size_h
    new_width = num_squares_w * square_size_w
    image = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(num_squares_h):
        for j in range(num_squares_w):
            start_x = j * square_size_w
            start_y = i * square_size_h
            color = 255 if (i + j) % 2 == 0 else 200
            image[
                start_y : start_y + square_size_h, start_x : start_x + square_size_w
            ] = color

    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


def run_vit_matte(input_x, masks):

    ### TODO !!!!!!! mask -> trimap
    """
    masks
        shape : (1, 1, 544, 890)
        dtype : bool
    """
    mask = masks.astype(np.uint8) * 255
    # mask = masks
    erode_kernel_size = 1
    dilate_kernel_size = 1
    trimap = generate_trimap(mask, erode_kernel_size, dilate_kernel_size).astype(
        np.float32
    )
    # return trimap
    trimap[trimap == 128] = 0.5
    trimap[trimap == 255] = 1

    # Vit Matte调用
    """
        input_x
            shape : (544, 890, 3)
            dtype : uint8
            type : numpy.ndarray
        trimap
            shape : (544, 890)
            dtype : float32
            max : 0.5
            min : 0.0
            type : numpy.ndarray
    """
    input = {
        "image": torch.from_numpy(input_x).permute(2, 0, 1).unsqueeze(0)
        / 255,  # norm to [0, 1]
        "trimap": torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0),
    }
    # import pdb; pdb.set_trace()
    torch.cuda.empty_cache()
    alpha = vitmatte(input)["phas"].flatten(0, 2)
    alpha = alpha.detach().cpu().numpy()

    return alpha

    # # get a green background
    # background = generate_checkerboard_image(input_x.shape[0], input_x.shape[1], 8)

    # # calculate foreground with alpha blending
    # foreground_alpha = input_x * np.expand_dims(alpha, axis=2).repeat(3,2)/255 + background * (1 - np.expand_dims(alpha, axis=2).repeat(3,2))/255

    # return foreground_alpha
    # background = generate_checkerboard_image(input_x.shape[0], input_x.shape[1], 8)

    # # calculate foreground with mask
    # foreground_mask = input_x * np.expand_dims(mask/255, axis=2).repeat(3,2)/255 + background * (1 - np.expand_dims(mask/255, axis=2).repeat(3,2))/255
    # return foreground_mask


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

vitmatte_models = {
    "vit_b": "./pretrained/ViTMatte_B_DIS.pth",
}

vitmatte_config = {
    "vit_b": "./configs/matte_anything.py",
}

vitmatte_model = "vit_b"
vitmatte = init_vitmatte(vitmatte_model)

matte_anything_path = "//home/fengkaige/codespace/Segment-and-Track-Anything"
os.chdir(matte_anything_path)
sys.path.append("./")


# choose good parameters in sam_args based on the first frame segmentation result
# other arguments can be modified in model_args.py
# note the object number limit is 255 by default, which requires < 10GB GPU memory with amp
sam_args["generator_args"] = {
    "points_per_side": 30,
    "pred_iou_thresh": 0.8,
    "stability_score_thresh": 0.9,
    "crop_n_layers": 1,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 200,
}

# Set Text args
"""
parameter:
    grounding_caption: Text prompt to detect objects in key-frames
    box_threshold: threshold for box
    text_threshold: threshold for label(text)
    box_size_threshold: If the size ratio between the box and the frame is larger than the box_size_threshold, the box will be ignored. This is used to filter out large boxes.
    reset_image: reset the image embeddings for SAM
"""
grounding_caption = "car.suv"
# grounding_caption = "door"
# grounding_caption = "girl"
# grounding_caption = "Beverage bottles"
# grounding_caption = ""
# grounding_caption = "cups"
# grounding_caption = "left cups"
box_threshold, text_threshold, box_size_threshold, reset_image = 0.35, 0.5, 0.5, True

cap = cv2.VideoCapture(io_args["input_video"])
frame_idx = 0
segtracker = SegTracker(segtracker_args, sam_args, aot_args)
segtracker.restart_tracker()
with torch.cuda.amp.autocast():
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # pred_mask, annotated_frame = segtracker.detect_and_seg_vitmatte(
        #     frame,
        #     grounding_caption,
        #     box_threshold,
        #     text_threshold,
        #     box_size_threshold,
        #     False,
        #     run_vit_matte,
        # )
        pred_mask, annotated_frame = segtracker.detect_and_seg(
            frame,
            grounding_caption,
            box_threshold,
            text_threshold,
            box_size_threshold,
            False,
        )

        torch.cuda.empty_cache()
        obj_ids = np.unique(pred_mask)
        obj_ids = obj_ids[obj_ids != 0]
        print(
            "processed frame {}, obj_num {}".format(frame_idx, len(obj_ids)), end="\n"
        )

        # Vit Matte
        # pred_mask = run_vit_matte(frame, pred_mask)

        break
    cap.release()
    init_res = draw_mask(annotated_frame, pred_mask, id_countour=False)
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(init_res)
    plt.show()
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(colorize_mask(pred_mask))
    plt.show()

    del segtracker
    torch.cuda.empty_cache()
    gc.collect()
