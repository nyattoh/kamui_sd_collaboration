import os
import gradio as gr
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from PIL import Image
import numpy as np

# CUDA設定
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ControlNetモデルの初期化
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

# パイプラインの初期化
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to(device)

def adjust_size(width, height, max_size=1024, min_size=256):
    """画像サイズを調整する（アスペクト比を保持）"""
    # アスペクト比を計算
    aspect = width / height
    
    # 最大サイズを超える場合は縮小
    if width > max_size or height > max_size:
        if width > height:
            width = max_size
            height = int(width / aspect)
        else:
            height = max_size
            width = int(height * aspect)
    
    # 最小サイズより小さい場合は拡大
    if width < min_size or height < min_size:
        if width < height:
            width = min_size
            height = int(width / aspect)
        else:
            height = min_size
            width = int(height * aspect)
    
    # 8の倍数に調整（Stable Diffusionの要件）
    width = (width // 8) * 8
    height = (height // 8) * 8
    
    return width, height

def process_image(image):
    """入力画像を前処理"""
    try:
        from controlnet_aux import CannyDetector
        
        # Cannyエッジ検出器の初期化
        canny = CannyDetector()
        
        # 画像をNumPy配列に変換
        if isinstance(image, str):
            image = Image.open(image)
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # 画像サイズを取得と調整
        height, width = image.shape[:2]
        width, height = adjust_size(width, height)
        
        # 必要に応じてリサイズ
        if (width, height) != (image.shape[1], image.shape[0]):
            image = Image.fromarray(image).resize((width, height))
            image = np.array(image)
        
        # Cannyエッジ検出を適用
        control_image = canny(image, low_threshold=100, high_threshold=200)
        
        return control_image, width, height
    except Exception as e:
        print(f"画像処理中にエラーが発生しました: {e}")
        raise

def generate_image(input_image, prompt, negative_prompt="", num_steps=30, guidance_scale=7.5):
    try:
        if input_image is None:
            raise ValueError("入力画像が必要です")
        
        # 入力画像の前処理とサイズ取得
        control_image, img_width, img_height = process_image(input_image)
        
        # 画像をPIL Imageに変換
        if isinstance(control_image, np.ndarray):
            control_image = Image.fromarray(control_image)
        
        print(f"生成サイズ: {img_width}x{img_height}")
        
        # 画像生成
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=img_width,
                height=img_height,
                controlnet_conditioning_scale=1.0
            ).images[0]
        
        return image
    except Exception as e:
        print(f"画像生成中にエラーが発生しました: {e}")
        return None

# Gradioインターフェースの作成
with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown("# KAMUI Image Generator")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="入力画像", type="numpy", interactive=True)
            prompt = gr.Textbox(label="プロンプト", placeholder="ごじにゃキャラクター, かわいい, 高品質", interactive=True)
            negative_prompt = gr.Textbox(label="ネガティブプロンプト", value="worst quality, low quality, normal quality", interactive=True)
            
            with gr.Row():
                steps = gr.Slider(minimum=1, maximum=150, value=30, step=1, label="ステップ数", interactive=True)
                guidance = gr.Slider(minimum=1, maximum=20, value=7.5, step=0.5, label="CFGスケール", interactive=True)
            
            generate_btn = gr.Button("画像を生成", variant="primary")
        
        with gr.Column():
            output = gr.Image(label="生成された画像")
            error_output = gr.Markdown(visible=False)
    
    # エラーハンドリング
    def on_error(e):
        return None, gr.update(visible=True, value=f"エラーが発生しました: {str(e)}")
    
    # イベントの設定
    generate_btn.click(
        fn=generate_image,
        inputs=[input_image, prompt, negative_prompt, steps, guidance],
        outputs=[output, error_output],
        api_name=None
    )

# サーバーの起動
if __name__ == "__main__":
    demo.queue(max_size=1).launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        show_error=True,
        debug=True
    )
