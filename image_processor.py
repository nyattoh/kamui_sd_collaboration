import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from rembg import remove
import logging
import traceback

logger = logging.getLogger("KAMUI.ImageProcessor")

class ImageSeparator:
    def __init__(self):
        self.segmenter = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.segmenter.to(self.device)
        logger.info(f"ImageSeparator初期化完了: device={self.device}")

    def _ensure_pil_image(self, image):
        """入力が確実にPIL Imageオブジェクトであることを確認"""
        if isinstance(image, Image.Image):
            return image
        
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image, 'L')
            elif image.ndim == 3:
                if image.shape[2] == 3:
                    return Image.fromarray(image, 'RGB')
                elif image.shape[2] == 4:
                    return Image.fromarray(image, 'RGBA')
                elif image.shape[2] == 1:
                    return Image.fromarray(image[:,:,0], 'L')
        
        raise TypeError(f"サポートされていない画像形式: {type(image)}")

    def separate_character(self, image: Image.Image) -> tuple[Image.Image, Image.Image]:
        """キャラクターと背景を分離する"""
        try:
            # 入力画像のチェックと変換
            logger.debug(f"separate_character: 入力画像タイプ={type(image)}, サイズ={getattr(image, 'size', 'unknown')}")
            pil_image = self._ensure_pil_image(image)
            
            # RGB画像に変換（rembgはRGBA形式を想定）
            if pil_image.mode != 'RGB' and pil_image.mode != 'RGBA':
                logger.debug(f"画像モードを変換: {pil_image.mode} -> RGB")
                pil_image = pil_image.convert('RGB')
            
            # キャラクター抽出
            logger.debug("rembg処理開始")
            try:
                character_mask = remove(pil_image)
                logger.debug(f"rembg処理完了: 結果タイプ={type(character_mask)}")
            except Exception as e:
                logger.error(f"rembg処理エラー: {str(e)}\n{traceback.format_exc()}")
                raise ValueError(f"キャラクター抽出に失敗しました: {str(e)}")
            
            # NumPy配列に変換してPIL Imageを生成
            if not isinstance(character_mask, Image.Image):
                logger.debug(f"character_maskをPIL Imageに変換: タイプ={type(character_mask)}")
                if isinstance(character_mask, np.ndarray):
                    character = Image.fromarray(character_mask)
                else:
                    raise TypeError(f"rembgの出力が未知の型です: {type(character_mask)}")
            else:
                character = character_mask
                
            logger.debug(f"キャラクター画像: サイズ={character.size}, モード={character.mode}")
            
            # 背景抽出（元画像からキャラクターを除去）
            original_array = np.array(pil_image)
            character_array = np.array(character)
            
            # 次元とサイズの確認
            logger.debug(f"元画像配列: shape={original_array.shape}")
            logger.debug(f"キャラクター配列: shape={character_array.shape}")
            
            # 背景作成
            background_array = original_array.copy()
            
            # キャラクターに対応するピクセルをマスクする
            if character_array.shape[2] == 4:  # アルファチャンネルがある場合
                logger.debug("アルファチャンネルを使用したマスク処理")
                alpha_mask = character_array[:, :, 3] > 0
                for i in range(3):  # RGB各チャンネルにマスクを適用
                    background_array[:, :, i][alpha_mask] = 0
            
            background = Image.fromarray(background_array)
            logger.debug(f"背景画像: サイズ={background.size}, モード={background.mode}")
            
            return character, background
            
        except Exception as e:
            logger.error(f"キャラクター分離エラー: {str(e)}\n{traceback.format_exc()}")
            raise

    def preserve_style(self, source: Image.Image, target: Image.Image) -> Image.Image:
        """画像のスタイルを保持しながら変換する"""
        try:
            # 入力チェック
            source_pil = self._ensure_pil_image(source)
            target_pil = self._ensure_pil_image(target)
            
            logger.debug(f"スタイル保持: ソース={source_pil.size}, モード={source_pil.mode}, ターゲット={target_pil.size}, モード={target_pil.mode}")
            
            # アルファチャンネルの有無を確認
            has_alpha = 'A' in target_pil.mode
            
            # RGB画像に変換してスタイル処理を行う
            source_rgb = source_pil.convert('RGB')
            target_rgb = target_pil.convert('RGB')
            
            # セグメンテーションマップの取得
            inputs = self.processor(images=source_rgb, return_tensors="pt").to(self.device)
            outputs = self.segmenter(**inputs)
            seg_map = outputs.logits.argmax(dim=1)[0]
            
            # スタイル転送（シンプルな色調整で代用）
            source_array = np.array(source_rgb)
            target_array = np.array(target_rgb)
            
            # 色調とコントラストを調整（RGBチャンネルのみ）
            mean_source = np.mean(source_array, axis=(0,1))
            std_source = np.std(source_array, axis=(0,1))
            mean_target = np.mean(target_array, axis=(0,1))
            std_target = np.std(target_array, axis=(0,1))
            
            # 0除算を防ぐ
            std_target = np.where(std_target < 0.1, 0.1, std_target)
            
            # RGB値の正規化
            normalized_rgb = ((target_array - mean_target) / std_target) * std_source + mean_source
            normalized_rgb = np.clip(normalized_rgb, 0, 255).astype(np.uint8)
            
            # 結果画像の作成
            if has_alpha:
                # 元の画像からアルファチャンネルを抽出
                logger.debug("アルファチャンネルを保持して処理")
                alpha = np.array(target_pil.getchannel('A'))
                
                # RGBとアルファチャンネルを結合
                rgba = np.dstack((normalized_rgb, alpha))
                result = Image.fromarray(rgba, 'RGBA')
            else:
                result = Image.fromarray(normalized_rgb, 'RGB')
            
            logger.debug(f"スタイル保持処理完了: 結果サイズ={result.size}, モード={result.mode}")
            return result
            
        except Exception as e:
            logger.error(f"スタイル保持エラー: {str(e)}\n{traceback.format_exc()}")
            raise

    def process_image(self, image: Image.Image) -> tuple[Image.Image, Image.Image]:
        """画像を処理して、スタイルを保持したキャラクターと背景を返す"""
        try:
            logger.info("画像処理開始")
            character, background = self.separate_character(image)
            logger.debug("キャラクターと背景の分離完了")
            
            styled_character = self.preserve_style(image, character)
            logger.debug("キャラクターのスタイル保持処理完了")
            
            styled_background = self.preserve_style(image, background)
            logger.debug("背景のスタイル保持処理完了")
            
            logger.info("画像処理完了")
            return styled_character, styled_background
            
        except Exception as e:
            logger.error(f"画像処理エラー: {str(e)}\n{traceback.format_exc()}")
            raise
