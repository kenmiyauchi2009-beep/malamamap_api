"""BioCLIP 2.5 Huge 植物分類 API（CPU 推論専用版）。

既存の api.py（GPU/自動判定）はそのまま残し、こちらは GPU の無いサーバー
（さくらのクラウド 標準サーバー等）へのデプロイ用に、デバイスを CPU に固定した版。

- テキスト埋め込みは main.py と同じキャッシュ（txt_emb_cache.pt）を共有。
  キャッシュ生成済みなら起動時の重い埋め込み計算をスキップできる。
- CPU では画像1枚のエンコードに数秒かかる（GPU 版より遅い）。低トラフィック
  なデモ／検証用途を想定。

起動例（Ubuntu / Linux）:
    # HTTPS 終端は前段の Caddy が担当するため localhost のみで待ち受ける。
    # 外部公開は Caddy(443) 経由: https://<サーバ>/ でアクセスする。
    .venv/bin/uvicorn api_cpu:app --host 127.0.0.1 --port 8000
または:
    .venv/bin/python api_cpu.py
仮想環境を有効化してから起動する場合:
    source .venv/bin/activate
    uvicorn api_cpu:app --host 127.0.0.1 --port 8000

（Caddy を使わず uvicorn を直接外部公開する場合のみ --host 0.0.0.0 にする）

起動例（Windows）:
    .venv\\Scripts\\uvicorn api_cpu:app --host 0.0.0.0 --port 8000

エンドポイント・パラメータは api.py と同一:
    GET  /            … ブラウザで試せる簡易アップロードフォーム
    GET  /health      … 稼働確認（モデル/デバイス/候補種数）
    POST /classify    … 画像を投げて種（または属）を予測
        file   画像ファイル（必須）
        rank   species / genus（既定 species）
        top_k  上位何件返すか（既定 5、1〜50）
        genus  候補をこの属に限定（カンマ区切りで複数可、任意）
"""
from __future__ import annotations

import io
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import torch

# ---- CPU 推論の並列度チューニング（モデルロード前に設定するのが望ましい） ----
# 物理コア数に合わせてスレッドを使い、CPU での ViT-H エンコードを少しでも速くする。
_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", os.cpu_count() or 4))
torch.set_num_threads(_NUM_THREADS)
os.environ.setdefault("OMP_NUM_THREADS", str(_NUM_THREADS))

import PIL.Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# main.py のローダ（モデル設定・候補読み込み・キャッシュ生成）を再利用
from main import load_candidates, build_classifier, MODEL_STR, CANDIDATE_CSV

# ----------------------------------------------------------------------------
# CPU に固定。モデルはプロセスに1つだけ常駐させる（ロードが重いため）。
# 推論はロックで直列化する（CPU 上でもモデル状態を共有するため）。
# ----------------------------------------------------------------------------
_DEVICE = "cpu"
_classifier = None          # CustomLabelsClassifier
_classes: list[str] = []    # 候補種（二名法）
_genus_of: list[str] = []   # 各候補の属名（= 二名法の先頭語）
_lock = threading.Lock()

_VALID_RANKS = {"species", "genus"}


def _genus(name: str) -> str:
    return name.split(" ", 1)[0] if name else ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier, _classes, _genus_of
    species = load_candidates(CANDIDATE_CSV)
    _classifier = build_classifier(species, _DEVICE)
    _classes = list(_classifier.classes)
    _genus_of = [_genus(c) for c in _classes]
    # ウォームアップ（初回推論のレイテンシを起動時に吸収）
    try:
        warm = PIL.Image.new("RGB", (224, 224), (0, 128, 0))
        _classify_image(warm, top_k=1, genera=[], rank="species")
    except Exception:
        pass
    yield
    _classifier = None


app = FastAPI(title="BioCLIP 2.5 植物分類 API (CPU)", version="2.0.0-cpu", lifespan=lifespan)

# ブラウザ（file:// や localhost の別ポート）から fetch できるよう CORS を許可。
# ローカル/デモ用途のため全オリジン許可。公開時は allow_origins を絞ること。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_rank(rank: str) -> str:
    key = (rank or "species").strip().lower()
    if key not in _VALID_RANKS:
        raise HTTPException(
            status_code=422,
            detail=f"rank は {sorted(_VALID_RANKS)} のいずれか。受け取った値: {rank!r}",
        )
    return key


def _split_csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


@torch.no_grad()
def _classify_image(image: PIL.Image.Image, top_k: int, genera: list[str], rank: str) -> dict:
    """画像を分類。genus 指定があれば候補をその属に絞る（キャッシュ済み埋め込みを流用）。

    戻り値: {"candidate_taxa": int, "predictions": [{"name","genus","score"}, ...]}
    """
    with _lock:
        # 候補のサブセット選択（属フィルタ）
        if genera:
            genera_set = set(genera)
            idx = [i for i, g in enumerate(_genus_of) if g in genera_set]
            if not idx:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された属が候補に存在しません: {genera}",
                )
        else:
            idx = list(range(len(_classes)))

        txt = _classifier.txt_embeddings[:, idx]                 # (D, M)
        sub_classes = [_classes[i] for i in idx]
        sub_genus = [_genus_of[i] for i in idx]

        img_feat = _classifier.create_image_features([image])    # (1, D) 正規化済み
        probs = _classifier.create_probabilities(img_feat, txt)[0].cpu()  # (M,) サブセットで再正規化

    if rank == "genus":
        # 種スコアを属ごとに合算
        agg: dict[str, float] = {}
        for g, p in zip(sub_genus, probs.tolist()):
            agg[g] = agg.get(g, 0.0) + p
        ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        predictions = [{"name": g, "genus": g, "score": round(s, 6)} for g, s in ranked]
    else:  # species
        k = min(top_k, probs.shape[0])
        top = torch.topk(probs, k)
        predictions = [
            {
                "name": sub_classes[i],
                "genus": sub_genus[i],
                "score": round(float(p), 6),
            }
            for i, p in zip(top.indices.tolist(), top.values.tolist())
        ]

    return {"candidate_taxa": len(idx), "predictions": predictions}


@app.get("/health")
def health():
    ready = _classifier is not None
    info = {
        "status": "ok" if ready else "loading",
        "device": _DEVICE,
        "model": MODEL_STR,
        "num_threads": _NUM_THREADS,
    }
    if ready:
        info["num_candidates"] = len(_classes)
    return info


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    rank: str = Form("species"),
    top_k: int = Form(5),
    genus: Optional[str] = Form(None),
):
    if _classifier is None:
        raise HTTPException(status_code=503, detail="モデルをロード中です。少し待って再試行してください。")

    target_rank = _resolve_rank(rank)
    top_k = max(1, min(int(top_k), 50))
    genera = _split_csv(genus)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="空のファイルです。")
    try:
        image = PIL.Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="画像として読み込めませんでした。")

    try:
        result = _classify_image(image, top_k=top_k, genera=genera, rank=target_rank)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推論に失敗しました: {e}")

    return JSONResponse(
        {
            "filename": file.filename,
            "rank": target_rank,
            "filter": {"genus": genera, "candidate_taxa": result["candidate_taxa"]},
            "predictions": result["predictions"],
        }
    )


_FORM_HTML = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>BioCLIP 2.5 植物分類 API (CPU)</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:680px;margin:40px auto;padding:0 16px}
 label{display:block;margin:10px 0 4px;font-weight:600}
 input,button{padding:8px;font-size:15px}
 button{margin-top:16px;cursor:pointer}
 #out{white-space:pre-wrap;background:#f4f4f5;padding:14px;border-radius:8px;margin-top:20px}
 .row{display:flex;gap:16px}.row>div{flex:1}
</style></head><body>
<h1>🌿 BioCLIP 2.5 植物分類 API <small>(CPU)</small></h1>
<p>候補: ハワイの植物（hawaii_plants.csv）／ CPU 推論のため1枚あたり数秒かかります</p>
<form id="f">
  <label>画像ファイル</label>
  <input type="file" name="file" accept="image/*" required>
  <div class="row">
    <div><label>rank</label>
      <input name="rank" value="species" list="ranks">
      <datalist id="ranks"><option>species</option><option>genus</option></datalist></div>
    <div><label>top_k</label><input name="top_k" type="number" value="5" min="1" max="50"></div>
  </div>
  <div><label>genus（任意・カンマ区切り）</label><input name="genus" placeholder="例: Metrosideros"></div>
  <button type="submit">分類する</button>
</form>
<div id="out">結果がここに表示されます</div>
<script>
 const f=document.getElementById('f'),out=document.getElementById('out');
 f.addEventListener('submit',async e=>{
   e.preventDefault();out.textContent='推論中…';
   const data=new FormData(f);
   if(!data.get('genus'))data.delete('genus');
   try{
     const r=await fetch('/classify',{method:'POST',body:data});
     out.textContent=JSON.stringify(await r.json(),null,2);
   }catch(err){out.textContent='エラー: '+err;}
 });
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return _FORM_HTML


if __name__ == "__main__":
    import uvicorn

    # HTTPS 終端は前段の Caddy が担当し、外部公開は 443 経由に一本化する。
    # バックエンドは localhost のみで待ち受け、8000 番を外部に晒さない。
    uvicorn.run(app, host="127.0.0.1", port=8000)
