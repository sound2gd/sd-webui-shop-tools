import gradio as gr
import os
import torch
from modules import script_callbacks, scripts
from modules.safe import unsafe_torch_load, load
import modules.shared as shared
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict
from huggingface_hub import hf_hub_download


from PIL import Image


current_extension_directory = scripts.basedir()
sam_model_dir=os.path.join(current_extension_directory, "models")
model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(
    os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt']
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

class ShopSAMPredictor:
    def __init__(self):
        self.predictor = None
        self.is_loaded = False

    def load_sam_model(self, sam_checkpoint=sam_checkpoint):
        if self.is_loaded:
            return

        model_type = '_'.join(sam_checkpoint.split('_')[1:-1])
        sam_checkpoint = os.path.join(sam_model_dir, sam_checkpoint)
        torch.load = unsafe_torch_load
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        torch.load = load
        self.is_loaded = True

    def get_predictor(self):
        if not self.is_loaded:
            self.load_sam_model()
        return self.predictor

sam_holder = ShopSAMPredictor()
# sam = sam_model_registry[model_type](checkpoint=os.path.join(sam_model_dir, sam_checkpoint))
# sam.to(device=device)


def pil_resize_image(image: Image.Image, length=768):
    w, h = image.size
    # print("ori-size: ", (w,h))
    corp = (0, 0, w, h)
    if w > h:
        h = int((length * h / w))
        w = length
        ah = int(h / 8.0) * 8
        corp = (0, int((h - ah) / 2), w, ah + int((h - ah) / 2))
    elif w < h:
        w = int((length * w / h))
        h = length
        aw = int(w / 8.0) * 8
        corp = (int((w - aw) / 2), 0, int((w - aw) / 2) + aw, h)
    else:
        w = h = length
    rtn = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    if w % 8 != 0 or h % 8 != 0:
        rtn = rtn.crop(corp)
    return rtn


def resize_images_from_dir(files, path: str, save_path: str, length=512):
    for f in [x for x in files if not x.startswith(".")]:
        im = Image.open(os.path.join(path, f)).convert("RGB")
        im2 = pil_resize_image(im, length=length)
        im2.save(f'{save_path}/resize_{f}')

# os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

def load_model_hf(repo_id, filename, ckpt_config_filename, device=device):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("shop_tools: Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)


BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.25

def save_transparent_image(image_source, mask_tensor, save_path):
    save_path = f'{save_path}.png'
    if os.path.exists(save_path):
        print(f"shop_tools: 已存在{save_path}, 跳过处理")
        return
    # 确保掩码是二值化的（0和1）
    binary_mask = (mask_tensor > 0.5).astype(np.uint8)

    # 创建一个 RGBA 格式的新图像，初始时是全透明的
    transparent_image = np.zeros((*image_source.shape[:2], 4), dtype=np.uint8)

    # 将原始图像复制到新图像的 RGB 通道中
    transparent_image[:, :, :3] = image_source
    # cv2是BGR
    # transparent_image[:, :, :3] = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)

    # 根据掩码设置 Alpha 通道的值
    transparent_image[:, :, 3] = binary_mask * 255  # Alpha 通道，0 表示完全透明，255 表示完全不透明

    # 将带 Alpha 通道的图像保存为 PNG
    # cv2.imwrite(save_path, transparent_image)
    transparent_image_pil = Image.fromarray(transparent_image, 'RGBA')
    transparent_image_pil.save(f'{save_path}.png')


def label_prompt(local_image_path: str, out_image_path: str, prompt: str, box_threshold=0.3,text_threshold=0.25, save_mask=False, rembg=True):
    skip_mask = not save_mask
    skip_rembg = not rembg
    out_mask_file_path = f'{out_image_path}/mask_{os.path.basename(local_image_path)}'
    if save_mask and os.path.exists(out_mask_file_path):
        print(f"shop_tools: 已存在{out_mask_file_path}, 跳过处理")
        skip_mask = True
    out_rembg_file_path = f'{out_image_path}/rembg_{os.path.basename(local_image_path)}'
    if rembg and os.path.exists(out_rembg_file_path):
        print(f"shop_tools: 已存在{out_rembg_file_path}, 跳过处理")
        skip_rembg = True
    if skip_rembg and skip_mask:
        return

    print(f"shop_tools: prompt={prompt}, box={box_threshold}, text={text_threshold}")
    image_source, image = load_image(f'{local_image_path}')

    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    predictor=sam_holder.get_predictor()
    predictor.set_image(image_source)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    image_mask = masks[0][0].cpu().numpy()
    if save_mask:
        image_mask_pil = Image.fromarray(image_mask)
        image_mask_pil.save(out_mask_file_path)
    if rembg:
        save_transparent_image(image_source, image_mask, out_rembg_file_path)


dir_names = ["resize","rembg", "mask"]
def processing(input_dir, rembg_enabled, rembg_seg_prompt, rembg_box_threshold, rembg_text_threshold,
                    mask_enabled, mask_seg_prompt, mask_box_threshold, mask_text_threshold):
    if not os.path.exists(input_dir):
        return "原图文件夹不存在"
    # 创建3个文件夹
    for dir in dir_names:
        full_dir=os.path.join(input_dir, dir)
        if not os.path.exists(full_dir):
            os.makedirs(full_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if not f.startswith(tuple(dir_names))]
    full_dir_names = [os.path.join(input_dir, f) for f in dir_names]
    # 改编大小
    resize_images_from_dir(files, input_dir, full_dir_names[0])
    resize_dir=full_dir_names[0]
    if rembg_enabled:
        # 从大小改编后的文件夹移除背景
        for x in [os.path.join(resize_dir, f) for f in os.listdir(resize_dir) if not f.startswith(".")]:
            print(f'shop_tools: rembg正在处理{x}')
            label_prompt(x, full_dir_names[1], rembg_seg_prompt, text_threshold=rembg_text_threshold, box_threshold=rembg_box_threshold, rembg=True, save_mask=False)
    if mask_enabled:
        # 从大小改编后的文件夹生成遮罩
        for x in [os.path.join(resize_dir, f) for f in os.listdir(resize_dir) if not f.startswith(".")]:
            print(f'shop_tools: mask正在处理{x}')
            label_prompt(x, full_dir_names[2], mask_seg_prompt, text_threshold=mask_text_threshold, box_threshold=mask_box_threshold, rembg=False, save_mask=True)
    return "操作完成"


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ShopTools:
        input_tab_state = gr.State(value=0)
        with gr.Row():
            with gr.Column():
                with gr.Tabs():
                    with gr.TabItem(label="文件夹批量处理") as input_tab_dir:
                        input_dir = gr.Textbox(label="原图文件夹", **shared.hide_dirs)
                        model_name = gr.Dropdown(label="SAM模型", elem_id="sam_model", choices=model_list,
                                                 value=model_list[0] if len(model_list) > 0 else None)
                with gr.Accordion("主体设置", open=True):
                    with gr.Tab("SAM & Groundino 参数"):
                        rembg_enabled = gr.Checkbox(label="启用", show_label=True, value=True)
                        rembg_seg_prompt = gr.Textbox(label = "groundino prompt", show_label=True, value="human and clothes")
                        rembg_box_threshold = gr.Slider(0, 1, value=0.3, step=0.01, label="box_threshold", show_label=True)
                        rembg_text_threshold = gr.Slider(0, 1, value=0.25, step=0.01, label="text_threshold", show_label=True)
                with gr.Accordion("遮罩设置", open=True):
                    with gr.Tab("SAM & Groundino 参数"):
                        mask_enabled = gr.Checkbox(label="启用", show_label=True, value=True)
                        mask_seg_prompt = gr.Textbox(label = "groundino prompt", show_label=True, value="clothes only")
                        mask_box_threshold = gr.Slider(0, 1, value=0.3, step=0.01, label="box_threshold", show_label=True)
                        mask_text_threshold = gr.Slider(0, 1, value=0.25, step=0.01, label="text_threshold", show_label=True)

                submit = gr.Button(value="开始处理")
            with gr.Row():
                with gr.Column():
                    gallery = gr.Textbox(label="outputs", show_label=False, elem_id="gallery")
                    # gallery = gr.Gallery(label="outputs", show_label=True, elem_id="gallery").style(grid=2, object_fit="contain")

        # 0: single 1: batch 2: batch dir
        # input_tab_single.select(fn=lambda: 0, inputs=[], outputs=[input_tab_state])
        # input_tab_batch.select(fn=lambda: 1, inputs=[], outputs=[input_tab_state])
        input_tab_dir.select(fn=lambda: 0, inputs=[], outputs=[input_tab_state])
        submit.click(
            processing,
            inputs=[input_dir, rembg_enabled, rembg_seg_prompt, rembg_box_threshold, rembg_text_threshold,
                    mask_enabled, mask_seg_prompt, mask_box_threshold, mask_text_threshold],
            outputs=gallery
        )

    return [(ShopTools, "电商插件", "shop_tools")]

script_callbacks.on_ui_tabs(on_ui_tabs)
