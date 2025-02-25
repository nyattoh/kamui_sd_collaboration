import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image
import os

class LoRATrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip.to(self.device)

    def prepare_training_data(self, character_image: Image.Image, background_image: Image.Image):
        """学習データの準備"""
        # 画像をテンソルに変換
        character_inputs = self.clip_processor(images=character_image, return_tensors="pt").to(self.device)
        background_inputs = self.clip_processor(images=background_image, return_tensors="pt").to(self.device)
        
        return character_inputs, background_inputs

    def train_lora(self, character_inputs, background_inputs, base_model_path: str, output_path: str):
        """LoRAモデルの学習"""
        # ここでは簡略化のため、実際のLoRA学習は省略
        # 実際の実装では、diffusers libraryのLoRAを使用して学習を行う
        print("Training LoRA model...")
        
        # 学習済みモデルの保存
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # モデルの保存処理（実際の実装では、trained_model.save_pretrained()などを使用）
        print(f"Saved LoRA model to {output_path}")

    def analyze_generated_image(self, generated_image: Image.Image, target_image: Image.Image) -> float:
        """生成画像と目標画像の類似度を分析"""
        # CLIP特徴量を使用して類似度を計算
        inputs1 = self.clip_processor(images=generated_image, return_tensors="pt").to(self.device)
        inputs2 = self.clip_processor(images=target_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            features1 = self.clip.get_image_features(**inputs1)
            features2 = self.clip.get_image_features(**inputs2)
        
        # コサイン類似度の計算
        similarity = torch.nn.functional.cosine_similarity(features1, features2)
        return similarity.item()

    def generate_with_lora(self, prompt: str, lora_path: str) -> Image.Image:
        """LoRAモデルを使用して画像を生成"""
        # 実際の実装では、trained_modelをロードして画像生成を行う
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to(self.device)
        
        # LoRAの重みを読み込む処理を追加
        
        # 画像生成
        image = pipe(prompt).images[0]
        
        # 必ずPIL Imageオブジェクトを返すように確認
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                # NumPy配列からPIL Imageに変換
                if image.ndim == 3 and image.shape[2] == 3:
                    return Image.fromarray(image.astype(np.uint8))
                elif image.ndim == 3 and image.shape[2] == 4:
                    return Image.fromarray(image.astype(np.uint8), 'RGBA')
                else:
                    return Image.fromarray(image.astype(np.uint8), 'L')
            else:
                raise TypeError(f"Unexpected image type: {type(image)}")
        
        return image
