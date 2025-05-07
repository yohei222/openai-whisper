import whisper
import time  # 時間計測のためのモジュールを追加

def transcribe_audio(audio_file_path):
    """
    指定された音声ファイルをWhisperでテキストに変換します。

    Args:
        audio_file_path (str): 音声ファイルのパス。

    Returns:
        dict: 変換されたテキストとセグメント情報を含む辞書。エラーが発生した場合はNoneを返します。
    """
    try:
        start_time = time.time()  # 全体の処理開始時間
        
        # モデルロード時間の計測開始
        model_load_start = time.time()
        # より高精度な large モデルをロード (必要に応じて)
        # base モデルを使用するとより早くモデルのロードが終了する
        model = whisper.load_model("large")
        model_load_time = time.time() - model_load_start  # モデルロード時間
        print(f"Large モデルのロードが完了しました。所要時間: {model_load_time:.2f}秒")
        
        # 文字起こし処理時間の計測開始
        transcribe_start = time.time()
        result = model.transcribe(audio_file_path, language="ja")
        transcribe_time = time.time() - transcribe_start  # 文字起こし処理時間
        
        total_time = time.time() - start_time  # 全体の処理時間
        print(f"処理完了 - モデルロード: {model_load_time:.2f}秒, 文字起こし: {transcribe_time:.2f}秒, 合計: {total_time:.2f}秒")
        
        # テキストとセグメント情報の両方を返す
        return {
            "text": result["text"],
            "segments": result["segments"]
        }
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

def format_time(seconds):
    """
    秒数を「時:分:秒」形式に変換します。
    
    Args:
        seconds (float): 秒数
        
    Returns:
        str: 「時:分:秒」形式の文字列
    """
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

if __name__ == "__main__":
    total_start = time.time()  # メイン処理の開始時間
    
    audio_file = "sampleTokyo.wav"  # 処理したい音声ファイルのパスに置き換えてください
    result = transcribe_audio(audio_file)

    if result:
        print("------------テキスト化された内容（全体）------------")
        print(result["text"])
        
        print("\n------------セグメント単位の文字起こし結果------------")
        for i, segment in enumerate(result["segments"]):
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            text = segment["text"]
            print(f"[{start_time} → {end_time}] {text}")                    
    else:
        print("音声のテキスト化に失敗しました。")
    
    print(f"全体の実行時間: {time.time() - total_start:.2f}秒")