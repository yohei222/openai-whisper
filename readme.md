## Install Dependencies
`pip install openai-whisper`

`brew install ffmpeg`

## Execution/Output

```
yoheikuji@YoheinoMacBook-Pro openai-whisper % python transcribe.py
Large モデルのロードが完了しました。所要時間: 14.21秒
/Users/yoheikuji/.pyenv/versions/3.12.1/lib/python3.12/site-packages/whisper/transcribe.py:126: UserWarning: FP16 is not supported on CPU; using FP32 instead
  warnings.warn("FP16 is not supported on CPU; using FP32 instead")
処理完了 - モデルロード: 14.21秒, 文字起こし: 51.65秒, 合計: 65.86秒
------------テキスト化された内容（全体）------------
朝野智美です。今日の東京株式市場で日経平均株価は小幅促進となっています。終わり値は昨日に比べ22円72銭高の11,088円58銭でした。当初一部の値上がり銘柄数は1,146。対して値下がりは368。変わらずは104銘柄となっています。ここでプレゼントのお知らせです。この番組では毎月発行のマンスリーレポート4月号を抽選で10名様にプレゼントいたします。お申し込みはお電話で東京03-0107-8373、03-0107-8373まで。以上番組からのお知らせでした。

------------セグメント単位の文字起こし結果------------
[00:00:00.00 → 00:00:00.88] 朝野智美です。
[00:00:02.18 → 00:00:06.66] 今日の東京株式市場で日経平均株価は小幅促進となっています。
[00:00:07.38 → 00:00:14.76] 終わり値は昨日に比べ22円72銭高の11,088円58銭でした。
[00:00:16.26 → 00:00:19.60] 当初一部の値上がり銘柄数は1,146。
[00:00:20.34 → 00:00:22.84] 対して値下がりは368。
[00:00:23.54 → 00:00:25.82] 変わらずは104銘柄となっています。
[00:00:27.40 → 00:00:29.12] ここでプレゼントのお知らせです。
[00:00:29.12 → 00:00:35.76] この番組では毎月発行のマンスリーレポート4月号を抽選で10名様にプレゼントいたします。
[00:00:36.40 → 00:00:45.30] お申し込みはお電話で東京03-0107-8373、03-0107-8373まで。
[00:00:46.22 → 00:00:48.06] 以上番組からのお知らせでした。

------------セグメント情報の詳細（参考）------------
セグメント数: 10
セグメントの例（最初の1つ）: {'id': 0, 'seek': 0, 'start': 0.0, 'end': 0.88, 'text': '朝野智美です。', 'tokens': [50365, 46610, 37178, 5094, 118, 9175, 4767, 1543, 50409], 'temperature': 0.0, 'avg_logprob': -0.123349038485823, 'compression_ratio': 1.3576923076923078, 'no_speech_prob': 0.46981409192085266}
全体の実行時間: 66.15秒
```
