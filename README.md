# KAMUI Stable Diffusion Collaboration Image Generator

画像の背景とキャラクターを分離し、それぞれのスタイルを保持しながらLoRAモデルを学習・生成するAIイメージジェネレーター。
フィードバックループを使用して、生成画像の品質を段階的に向上させます。

## 必要要件

- Python 3.10以上
- CUDA対応のNVIDIA GPU（16GB VRAM推奨）
- Windows 10/11

## インストール方法

1. リポジトリをクローンまたはダウンロード：
```bash
git clone [repository-url]
cd kamui_sd_collaboration
```

2. Python仮想環境の作成（推奨）：
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. 必要なパッケージのインストール：
```bash
pip install -r requirements.txt
pip install onnxruntime-gpu
```

## 使用方法

1. アプリケーションの起動：
```bash
python app.py
```

2. ブラウザで表示されるインターフェースにアクセス
   - デフォルトでは `http://localhost:7860` で起動します

3. 操作手順：
   - 「入力画像」欄に処理したい画像をアップロード
   - 「プロンプト」欄に生成したい画像の説明を入力
   - 「生成開始」ボタンをクリックして処理を開始

## 主な機能

- **画像分離**: 背景とキャラクターを自動的に分離
- **スタイル保持**: 元画像のスタイルを保持しながら生成
- **LoRA学習**: カスタムLoRAモデルの自動学習
- **フィードバックループ**: 生成画像の品質を段階的に向上
- **Webインターフェース**: 直感的なGradioベースのUI

## ファイル構成

- `app.py`: メインアプリケーション（Gradioインターフェース）
- `image_processor.py`: 画像処理モジュール
- `lora_trainer.py`: LoRAモデルの学習と生成
- `feedback_loop.py`: フィードバックループの制御
- `requirements.txt`: 必要なPythonパッケージ一覧

## 注意事項

- 初回起動時はモデルのダウンロードに時間がかかる場合があります
- 画像生成には十分なGPUメモリが必要です
- 処理時間は入力画像のサイズやフィードバックループの回数により変動します

## エラー対応

よくあるエラーと解決方法：

1. CUDA関連のエラー
   - NVIDIA GPUドライバーが最新であることを確認
   - CUDA Toolkitがインストールされていることを確認

2. メモリ不足エラー
   - 入力画像のサイズを小さくする
   - 他のアプリケーションを終了してメモリを解放

3. モジュールが見つからないエラー
   - 仮想環境が有効になっていることを確認
   - `pip install -r requirements.txt`を再実行

## ライセンス

MITライセンス
