from image_processor import ImageSeparator
from lora_trainer import LoRATrainer
import torch
from PIL import Image
import os
from typing import Tuple, Optional
import numpy as np

class FeedbackLoop:
    def __init__(self, base_model_path: str, output_dir: str):
        self.separator = ImageSeparator()
        self.trainer = LoRATrainer()
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        self.current_iteration = 0
        self.max_iterations = 5
        self.similarity_threshold = 0.95

    def process_iteration(self, 
                         input_image: Image.Image,
                         prompt: str) -> Tuple[Image.Image, float]:
        """1回のフィードバックループを実行"""
        # 画像の分離
        character, background = self.separator.process_image(input_image)
        
        # 学習データの準備
        char_inputs, bg_inputs = self.trainer.prepare_training_data(character, background)
        
        # LoRAモデルの学習
        iteration_path = os.path.join(self.output_dir, f"iteration_{self.current_iteration}")
        self.trainer.train_lora(char_inputs, bg_inputs, self.base_model_path, iteration_path)
        
        # 新しい画像の生成
        generated_image = self.trainer.generate_with_lora(prompt, iteration_path)
        
        # 生成画像の評価
        similarity = self.trainer.analyze_generated_image(generated_image, input_image)
        
        return generated_image, similarity

    def run(self, input_image: Image.Image, prompt: str) -> Optional[Image.Image]:
        """フィードバックループを実行"""
        best_image = None
        best_similarity = 0.0
        
        while self.current_iteration < self.max_iterations:
            print(f"Starting iteration {self.current_iteration + 1}")
            
            generated_image, similarity = self.process_iteration(input_image, prompt)
            
            if similarity > best_similarity:
                best_image = generated_image
                best_similarity = similarity
            
            print(f"Iteration {self.current_iteration + 1} completed. Similarity: {similarity:.4f}")
            
            # 十分な類似度が得られた場合は終了
            if similarity >= self.similarity_threshold:
                print(f"Achieved target similarity threshold: {similarity:.4f}")
                break
            
            self.current_iteration += 1
        
        # 最終的な画像が確実にPIL Imageオブジェクトであることを確認
        if best_image is not None and not isinstance(best_image, Image.Image):
            if isinstance(best_image, np.ndarray):
                # NumPy配列からPIL Imageに変換
                if best_image.ndim == 3 and best_image.shape[2] == 3:
                    best_image = Image.fromarray(best_image.astype(np.uint8))
                elif best_image.ndim == 3 and best_image.shape[2] == 4:
                    best_image = Image.fromarray(best_image.astype(np.uint8), 'RGBA')
                else:
                    best_image = Image.fromarray(best_image.astype(np.uint8), 'L')
        
        return best_image

def create_image_with_feedback(input_image: Image.Image,
                             prompt: str,
                             base_model_path: str,
                             output_dir: str) -> Image.Image:
    """フィードバックループを使用して画像を生成する"""
    # フィードバックループの初期化と実行
    loop = FeedbackLoop(base_model_path, output_dir)
    result_image = loop.run(input_image, prompt)
    
    if result_image is None:
        raise RuntimeError("Failed to generate satisfactory image")
    
    return result_image
