import io
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from object_manager import ObjectManager

class Sam2_Manager():
    def __init__(self):
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
        self.device = device
        self.inference_state = None
        self.predictor = None

    def init_inference(self, input_path: str):
        sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        self.predictor  = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device = self.device)
        print('Build predictor Done')
        #print(predictor)
        self.inference_state =  self.predictor.init_state(video_path = input_path)
        #print(inference_state)
        print('inference state Done')
    
    def reset_init(self):
        self.predictor.reset_state(self.inference_state)
    
    def update_video_mask(self, mask_img, out_img, h: int, w: int, colour: list):
        for y in range(0, h):
            for x in range(0, w):
                if (mask_img[y, x, 0]):
                    # alpha 60% = 153
                    out_img[y, x] = colour + [153]
        return out_img

    def doImagePredic(self, frame_id: int, obj_prompts: dict, objMngr: ObjectManager):

        for prompt in obj_prompts.values():
            if not prompt.isActivate() or prompt.getFrameId() != frame_id:
                continue
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state = self.inference_state,
            frame_idx=frame_id,
            obj_id=prompt.object_id,
            points=prompt.input_position,
            labels=prompt.input_label,
            )

        print(out_obj_ids)


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
            out_img = self.update_video_mask(mask.reshape(h, w, 1), out_img, h, w, colour)

        img_file = io.BytesIO()
        plt.imsave(img_file, out_img, cmap = 'BrBG')

        return Image.open(img_file), h, w
    
    def doVideoPredic(self, obj_prompts: dict, start_frame_id: int, frame_len: int, objMngr: ObjectManager):

        for frame_id in range(start_frame_id, frame_len):        
            for prompt in obj_prompts.values():
                if not prompt.isActivate() or prompt.getFrameId() != frame_id:
                    continue
                print('add_new_points', 'obj_id:', prompt.object_id, prompt.input_position)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state = self.inference_state,
                frame_idx= frame_id,
                obj_id=prompt.object_id,
                points=prompt.input_position,
                labels=prompt.input_label,
                )
            
        mask =  (out_mask_logits[0] > 0.0).cpu().numpy()
        h, w = mask.shape[-2:]

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state, 
                                                                                             start_frame_idx = start_frame_id, 
                                                                                             max_frame_num_to_track = frame_len):
            video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        for out_idx in range(start_frame_id, frame_len):
            out_img = np.zeros((h, w, 4), np.uint8)
            for out_obj_id, out_mask in video_segments[out_idx].items():
                colour = objMngr.get_entity_colour(out_obj_id)
                if colour == None:
                    print("doVideoPredic, Object_{}".format(out_obj_id), "is not set")
                    continue
                height, width = out_mask.shape[-2:]
                out_img = self.update_video_mask(out_mask.reshape(h, w, 1), out_img, height, width, colour)
                plt.imsave('output/{}.png'.format(out_idx), out_img, cmap = 'BrBG')
        print('Tracking Done')
        return True
    
    def doVideoPredic_V2(self, h:int, w:int, start_frame_id: int, frame_len: int, objMngr: ObjectManager):
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        for out_idx in range(start_frame_id, frame_len):
            out_img = np.zeros((h, w, 4), np.uint8)
            for out_obj_id, out_mask in video_segments[out_idx].items():
                colour = objMngr.get_entity_colour(out_obj_id)
                if colour == None:
                    print("doVideoPredic, Object_{}".format(out_obj_id), "is not set")
                    continue
                height, width = out_mask.shape[-2:]
                out_img = self.update_video_mask(out_mask.reshape(h, w, 1), out_img, height, width, colour)
                plt.imsave('output/{}.png'.format(out_idx), out_img, cmap = 'BrBG')
        print('Tracking Done')
        return True