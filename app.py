import gradio as gr
from feedback_loop import create_image_with_feedback
import os
from PIL import Image
import numpy as np
import logging
import traceback
import sys

# より詳細なログ設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("KAMUI")

def debug_log_image_info(stage, img):
    """画像情報のデバッグログ出力"""
    try:
        if img is None:
            logger.debug(f"{stage}: 画像はNoneです")
            return
            
        if isinstance(img, Image.Image):
            logger.debug(f"{stage}: PIL Image, サイズ={img.size}, モード={img.mode}")
        elif isinstance(img, np.ndarray):
            logger.debug(f"{stage}: NumPy配列, shape={img.shape}, dtype={img.dtype}, min={np.min(img)}, max={np.max(img)}")
        else:
            logger.debug(f"{stage}: 未知の型: {type(img)}")
    except Exception as e:
        logger.error(f"{stage}でのログ出力エラー: {str(e)}")

def validate_image(input_image):
    """画像データの完全なバリデーション"""
    debug_log_image_info("検証前", input_image)
    
    if not isinstance(input_image, (Image.Image, np.ndarray)):
        raise ValueError(f"Invalid image type: {type(input_image)}")
    
    if isinstance(input_image, np.ndarray):
        if input_image.size == 0:
            raise ValueError("Empty image array")
        if input_image.ndim not in (2, 3):
            raise ValueError(f"Invalid array dimensions: {input_image.ndim}")
        if input_image.dtype not in (np.uint8, np.float32):
            raise ValueError(f"Unsupported dtype: {input_image.dtype}")
    
    debug_log_image_info("検証後", input_image)
    return input_image

def convert_to_pil(input_image):
    """堅牢な画像変換処理"""
    try:
        debug_log_image_info("変換前", input_image)
        
        if isinstance(input_image, Image.Image):
            debug_log_image_info("変換不要(既にPIL)", input_image)
            return input_image
        
        if input_image is None:
            raise ValueError("入力画像がNoneです")
            
        if not isinstance(input_image, np.ndarray):
            raise TypeError(f"サポートされていない入力タイプ: {type(input_image)}")
        
        # 空の配列チェック
        if input_image.size == 0:
            raise ValueError("空の画像配列")
            
        # 次元数チェック
        if input_image.ndim not in (2, 3):
            raise ValueError(f"無効な配列の次元数: {input_image.ndim}")
        
        # 正規化処理
        if input_image.dtype == np.float32 or input_image.dtype == np.float64:
            if np.max(input_image) <= 1.0:
                input_image = (input_image * 255).astype(np.uint8)
            else:
                input_image = input_image.astype(np.uint8)
        elif input_image.dtype != np.uint8:
            input_image = input_image.astype(np.uint8)
        
        pil_image = None
        # チャンネル処理
        if input_image.ndim == 2:
            pil_image = Image.fromarray(input_image, 'L')
        elif input_image.ndim == 3 and input_image.shape[2] == 4:
            pil_image = Image.fromarray(input_image, 'RGBA')
        elif input_image.ndim == 3 and input_image.shape[2] == 3:
            pil_image = Image.fromarray(input_image, 'RGB')
        elif input_image.ndim == 3 and input_image.shape[2] == 1:
            pil_image = Image.fromarray(input_image[:,:,0], 'L')
        else:
            raise ValueError(f"サポートされていないチャンネル数: {input_image.shape[2] if input_image.ndim == 3 else '不明'}")
        
        debug_log_image_info("変換後", pil_image)
        return pil_image
    
    except Exception as e:
        logger.error(f"変換エラー: {str(e)}\n{traceback.format_exc()}")
        raise ValueError(f"画像変換に失敗しました: {str(e)}")

def generate_image(input_image, prompt):
    """画像生成の主要関数"""
    if input_image is None:
        logger.warning("入力画像がNoneです")
        return None, "入力画像を選択してください。"
    
    try:
        logger.info(f"画像生成開始: 入力画像タイプ={type(input_image)}, プロンプト='{prompt}'")
        
        # 入力画像のバリデーションと変換
        logger.debug("入力画像のバリデーション開始")
        validated_image = validate_image(input_image)
        logger.debug("入力画像のPIL変換開始")
        pil_image = convert_to_pil(validated_image)
        debug_log_image_info("PILに変換後", pil_image)
        
        # 出力ディレクトリの作成
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"出力ディレクトリ作成: {output_dir}")
        
        # 画像生成の実行
        logger.info("フィードバックループによる画像生成開始")
        try:
            result = create_image_with_feedback(
                input_image=pil_image,
                prompt=prompt,
                base_model_path="runwayml/stable-diffusion-v1-5",
                output_dir=output_dir
            )
            logger.info("フィードバックループによる画像生成完了")
            debug_log_image_info("生成結果", result)
        except Exception as e:
            logger.error(f"画像生成処理中のエラー: {str(e)}\n{traceback.format_exc()}")
            raise
        
        # 返り値がNumPy配列の場合はPIL Imageに変換
        if result is not None and not isinstance(result, Image.Image):
            logger.warning(f"結果が予期せぬタイプです: {type(result)}")
            if isinstance(result, np.ndarray):
                logger.debug("NumPy配列からPIL Imageへの変換開始")
                result = convert_to_pil(result)
                debug_log_image_info("NumPy→PIL変換後", result)
            else:
                logger.error(f"不明な画像形式: {type(result)}")
                raise TypeError(f"不明な画像形式: {type(result)}")
        
        if result is None:
            logger.error("画像生成結果がNoneです")
            return None, "画像生成に失敗しました。"
            
        # PIL Imageであることを最終確認
        if not isinstance(result, Image.Image):
            logger.error(f"変換後も無効な画像形式です: {type(result)}")
            raise TypeError(f"変換後も無効な画像形式です: {type(result)}")
            
        logger.info("画像生成処理が正常に完了")
        return result, "生成が完了しました。"
    
    except Exception as e:
        logger.error(f"致命的なエラー: {traceback.format_exc()}")
        return None, f"エラーが発生しました: {str(e)}"

# Gradioインターフェースの作成
with gr.Blocks() as demo:
    gr.Markdown("# KAMUI Image Generator with Feedback Loop")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="入力画像", type="numpy")
            prompt = gr.Textbox(label="プロンプト", placeholder="画像生成のプロンプトを入力してください")
            generate_btn = gr.Button("生成開始")
        
        with gr.Column():
            output_image = gr.Image(label="生成結果")
            status_text = gr.Textbox(label="状態", interactive=False)
    
    generate_btn.click(
        fn=generate_image,
        inputs=[input_image, prompt],
        outputs=[output_image, status_text]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
