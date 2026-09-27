# 推論の高速化（出力保持型）

`InferenceRuntime.synthesize()` を、出力 WAV がビット単位で変わらない範囲で高速化した。
CUDA Graph は RF サンプラーだけに使い、参照音声キャッシュと条件の再利用は MeanFlow でも効く。
品質が変わり得る変更（step 数、BF16、padding の除去など）は入れていない。

## 変更内容

| 変更 | 場所 | 効果の出どころ |
|---|---|---|
| CUDA Graph による DiT forward の replay | `irodori_tts/cuda_graph.py`、`inference_runtime.py` | batch 1 の forward は kernel 起動待ちが支配的で、replay で約半分になる |
| 参照音声 latent のキャッシュ | `inference_runtime.py`（`ReferenceLatentCache`） | 同じ参照音声の読み込み・リサンプル・codec encode（1 回約 250ms）を省く |
| duration 予測で作った条件の再利用 | `inference_runtime.py`、`rf.py`、`meanflow.py` | `encode_conditions` の 2 回目（約 80〜90ms）を省く |
| サンプリングループ内の GPU 同期の除去 | `rf.py` | 毎 step の `t.item()` と `torch.full(..., t)` による同期をなくす |
| 末尾無音検出のベクトル化 | `inference_runtime.py`（`find_flattening_point`） | フレームごとの同期 2 回（約 600 フレームで約 100ms）をなくす |
| timestep 周波数表のキャッシュ | `model.py` | CUDA Graph の capture 中に禁止される CPU→GPU コピーをなくす |

上流に既にあった機能（cuDNN 優先の SDPA、FA3 の分岐、context K/V cache、attention plan の共有、BF16 オプション）は変更していない。

## 設定

| 項目 | 既定値 | 説明 |
|---|---|---|
| `RuntimeKey.cuda_graph` | `auto` | `auto` / `on` / `off`。`auto` は CUDA かつ `compile_model=False` のときに有効 |
| 環境変数 `IRODORI_CUDA_GRAPH` | 未設定 | `cuda_graph=auto` のときだけ参照する。`off` / `0` で無効、`on` / `1` で必須 |
| `RuntimeKey.reference_cache_max_entries` | `16` | 参照音声キャッシュの件数上限。`0` で無効 |
| `RuntimeKey.reference_cache_max_bytes` | `64 MiB` | 参照音声キャッシュのバイト上限（CPU メモリ） |

CUDA Graph は次の場合に自動で eager に戻る。

- CUDA 以外のデバイス、`compile_model=True`
- LoRA アダプタを一度でも読み込んだ runtime（読み込み時に Graph を破棄する）
- capture の失敗（そのキーはリクエストの終わりまで eager）
- 空きメモリが「入力テンソルの合計 + 全メモリの 15%」を下回るとき

## CUDA Graph の作り

- 毎 step 変わる入力（`x_t`、`t`、`delta_t`）だけを Graph 専用の static バッファに複製し、replay の前にコピーする。
- それ以外のテンソル（条件、マスク、context K/V cache）は呼び出し元のテンソルを Graph が直接読む。オブジェクトの同一性を Graph のキーに含めるので、形が同じで中身が違う入力（CFG の cond と uncond）は別の Graph になる。
- speaker K/V scaling のように context K/V cache をその場で書き換えても、次の replay は書き換え後の値を読む。eager と同じ結果になる。
- Graph はリクエストの終わりにすべて破棄する。セリフごとに長さが変わるため、持ち越しても再利用されにくく、VRAM だけを使うため。
- capture に失敗すると、次の kernel 起動で一度だけ報告されるエラーが残る。これを失敗直後に消費して、eager での継続を保証する。
- capture は `capture_error_mode="thread_local"` で行う。既定の `global` では、capture 中に別スレッドが kernel の起動や同期をすると、そのスレッドの処理が失敗した（実測）。

## 検証

| 検証 | 内容 | 結果 |
|---|---|---|
| 上流版サンプラーとの比較 | 89f9d8f の `rf.py` と、小さなランダム重みのモデルで CFG 3 方式、speaker K/V、rescale、sway、truncation、step 数を組み合わせる（CPU、CUDA、CUDA Graph） | 576 通りすべてビット一致 |
| 実モデルの回帰 | v4.1-Small で 20 ケース（`ref_embed`、caption のみ、候補 2 つ、秒数の手動指定、speaker K/V の境界、参照 2 本、short→…→short など）を、変更前・変更後・`IRODORI_CUDA_GRAPH=off` で比較 | 20 ケースすべて hash 一致 |
| 実運用に近い連続読み上げ | 長さの違う 10 行 × 2 周 | 20 回すべて hash 一致 |
| 最大入力 | 30 秒生成、参照音声約 100 秒、候補 2 つ | hash 一致、OOM なし |
| 単体テスト | `tests/`（CUDA がない環境では CUDA のテストを skip） | 101 件成功 |

## 結果（変更前を 1.00 とした相対値）

FP32、CUDA、参照音声あり、seed 固定。

| 指標 | 変更後 |
|---|---|
| 連続読み上げ 10 行の合計時間（1 周目 / 2 周目） | 0.58 / 0.57 |
| 同じ文の反復、短文（音声約 4 秒）の中央値 | 0.52 |
| 同じ文の反復、中文（約 8 秒） | 0.68 |
| 同じ文の反復、長文（約 27 秒） | 0.91 |
| 連続読み上げの peak allocated / peak reserved | 0.89 / 0.98 |
| 最大入力の peak allocated / peak reserved | 1.01 / 1.09 |

長文ほど効果が小さいのは、CFG を含む batch 3 の forward が FP32 の計算量で律速しているためと考えている。
中文の長さで測ると、batch 3 の forward は Graph にしても 52.8ms が 47.6ms になっただけだった（batch 1 は 49.8ms が 22.4ms）。

Graph をリクエストをまたいで持ち越す版も試した。同じ文の反復は 0.45 / 0.62 / 0.86 まで縮んだが、
最大入力の peak reserved が 1.43 になった。連続読み上げの時間はほぼ同じだったので、持ち越さない形を採った。

## 見送ったもの

| 候補 | 理由 |
|---|---|
| テキスト 256 / caption 512 の padding を実長まで詰める | 数学的には同じだが、forward の出力に数 e-6 の差が出てビット一致しない |
| BF16 / TF32 | 精度が変わる。品質評価を伴う別の判断にする |
| codec の weight norm の畳み込み | ビット一致するが、decode の短縮が約 3% にとどまる |
| speaker state のキャッシュ | 条件の再利用で speaker encoder の実行は 1 リクエスト 1 回になった。残りの時間は計測していない |
| 系列長の bucket 化 | 計測していない。padding の除去と同じく attention の計算範囲が変わるので、ビット一致は見込めない |

## 再現

```text
python benchmarks/inference_speed.py --ref-wav <参照音声> --public
python -m pytest tests -q
```

ベンチは、測りたい版の `irodori_tts` が import される環境で流す（このチェックアウトなら editable install か
`PYTHONPATH`）。どの版を測ったかは、出力 JSON の `environment.irodori_tts` で確かめられる。
変更前と比べるときは、上流版が import される環境で同じコマンドを流し、`runs[].hash` を突き合わせる。

完全な結果（デバイス名や絶対パスを含む）は `artifacts/local/` に書き出され、git の管理対象外になる。

## 残るリスク

- 効果は GPU、ドライバ、PyTorch の版、文の長さ、参照音声の長さで変わる。
- CUDA Graph はリクエストごとに capture する。長さが毎回変わる使い方で速くなることは確認したが、capture の費用は長さに依存する。
- 参照音声キャッシュの hit は 2 回目以降の話で、初回の生成は速くならない。
- `IRODORI_CUDA_GRAPH` の値は runtime の生成時に読む。変えたらサーバーを起動し直す。
- capture 中に別スレッドが PyTorch 既定の CUDA 乱数生成器を使うと、PyTorch がその乱数を拒否する。推論は runtime のロックで直列化され、サンプラーはリクエストごとの生成器を使うので、この経路では起きない。同じプロセスで別の CUDA 処理を並行させる場合は注意する。
