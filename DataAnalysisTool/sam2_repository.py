import io
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from object_manager import ObjectManager

def init():
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    return device

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))   
    
def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def show_masks2(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        # if len(scores) > 1:
        #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        break
    plt.savefig('output.png')
    #plt.show()
    plt.close()

def get_mask_image(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if not borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    # mask_file = io.BytesIO()
    plt.imsave('output.png', mask_image, cmap = 'BrBG')
    print(mask_image.shape)
    plt.close()
    return mask_image

def getMaskedImage(image, masks, scores, borders=True):
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_image = get_mask_image(mask, plt.gca(), borders=borders)
        break
        
    return mask_image


def doImagePredic(input_path: str, frame_id: int, obj_prompts: dict, objMngr: ObjectManager):
    device = init()
    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    print('Build predictor Done')
    #print(predictor)
    inference_state = predictor.init_state(video_path = input_path)
    #print(inference_state)
    print('inference state Done')

    for prompt in obj_prompts.values():
        if not prompt.isActivate():
            continue
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_id,
        obj_id=prompt.object_id,
        points=prompt.input_position,
        labels=prompt.input_label,
        )
    #print(a)
    print(out_obj_ids)
    #print(out_mask_logits)

    mask =  (out_mask_logits[0] > 0.0).cpu().numpy()
    h_t, w_t = mask.shape[-2:]
    out_img = np.zeros((h_t, w_t, 4), np.uint8)

    for i in range(0, len(out_obj_ids)):
        colour = objMngr.get_entity_colour(out_obj_ids[i])
        if colour == None:
            print("Object_{}".format(out_obj_ids[i]), "is not set")
            continue
        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
        h, w = mask.shape[-2:]
        out_img = update_video_mask_2(mask.reshape(h, w, 1), out_img, h, w, colour)

    img_file = io.BytesIO()
    plt.imsave(img_file, out_img, cmap = 'BrBG')

    return Image.open(img_file), predictor, inference_state, h, w

def getColour(obj_id:int, objMngr: ObjectManager):
    cmap = plt.get_cmap("tab10")
    cmap_idx = objMngr.get_entity_colour(obj_id)
    colour = np.array([*cmap(cmap_idx)[:3], 0.6])
    return colour

def update_video_mask(mask, frame_id, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_file = io.BytesIO()
    # plt.imsave('output/{}.png'.format(frame_id), mask_image, cmap = 'BrBG')
    plt.imsave(mask_file, mask_image, cmap = 'BrBG')
    print(mask_image.shape)
    # plt.close()
    return Image.open(mask_file)

def update_video_mask_2(mask_img, out_img, h: int, w: int, colour: list):
    for y in range(0, h):
        for x in range(0, w):
            if (mask_img[y, x, 0]):
                # alpha 60% = 153
                out_img[y, x] = colour + [153]
    return out_img

def viewPreview_deactivated(input_path: str, frame_id: int, obj_prompts: dict):
    device = init()
    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    print('Build predictor Done')
    #print(predictor)
    inference_state = predictor.init_state(video_path = input_path)
    #print(inference_state)
    print('inference state Done')

    for prompt in obj_prompts.values():
        if not prompt.isActivate():
            continue
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_id,
        obj_id=prompt.object_id,
        points=prompt.input_position,
        labels=prompt.input_label,
        )
    #print(a)
    print(out_obj_ids)
    print(out_mask_logits)

    blended_image = update_video_mask((out_mask_logits[0] > 0.0).cpu().numpy(), frame_id, obj_id=out_obj_ids[0])
    for i in range(1, len(out_obj_ids)):
        mask_image = update_video_mask((out_mask_logits[i] > 0.0).cpu().numpy(), frame_id, obj_id=out_obj_ids[i])
        blended_image = Image.blend(blended_image, mask_image, 0.5)

    return blended_image, predictor, inference_state


def doVideoPredic(predictor, inference_state, frame_len: int, h:int, w:int, objMngr: ObjectManager):
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

    for out_idx in range(0, frame_len):
        out_img = np.zeros((h, w, 4), np.uint8)
        for out_obj_id, out_mask in video_segments[out_idx].items():
            colour = objMngr.get_entity_colour(out_obj_id)
            if colour == None:
                print("doVideoPredic, Object_{}".format(out_obj_id), "is not set")
                continue
            height, width = out_mask.shape[-2:]
            out_img = update_video_mask_2(out_mask.reshape(h, w, 1), out_img, height, width, colour)
            plt.imsave('output/{}.png'.format(out_idx), out_img, cmap = 'BrBG')
    return True
        
        

def test():
    cmap = plt.get_cmap("tab10")
    cmap_idx = 12
    color = np.array([*cmap(cmap_idx)[:3], 0.6])
    print(color)
    print(type(color))

    
