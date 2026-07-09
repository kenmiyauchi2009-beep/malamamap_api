"""BioCLIP 2.5 Huge で、候補種リスト（既定: ハワイの植物）に対し画像を分類する。

テキスト埋め込みは初回だけ計算してキャッシュ（.pt）に保存。
2回目以降はキャッシュを読むだけなので高速。
"""
import csv
import hashlib
import os
import sys

import torch
from bioclip import CustomLabelsClassifier

# ===== 設定 ===============================================================
MODEL_STR = "hf-hub:imageomics/bioclip-2.5-vith14"
CANDIDATE_CSV = "hawaii_plants.csv"   # 1列目見出し "species" の候補種リスト
CACHE_PATH = "txt_emb_cache.pt"       # テキスト埋め込みのキャッシュ
TOP_K = 5
# =========================================================================


def load_candidates(csv_path: str) -> list:
    with open(csv_path, encoding="utf-8") as fd:
        reader = csv.DictReader(fd)
        species = [row["species"].strip() for row in reader if row.get("species", "").strip()]
    return sorted(set(species))


def cache_key(species: list, model_str: str) -> str:
    h = hashlib.sha256(("\n".join(species) + "|" + model_str).encode("utf-8"))
    return h.hexdigest()


def build_classifier(species: list, device: str) -> CustomLabelsClassifier:
    key = cache_key(species, MODEL_STR)

    # キャッシュが使えるなら、埋め込み計算をスキップして即構築
    if os.path.exists(CACHE_PATH):
        cache = torch.load(CACHE_PATH, map_location=device)
        if cache.get("key") == key:
            print(f"キャッシュ利用: {CACHE_PATH}（{len(cache['classes'])} 種）")
            # ダミー1種で初期化して重い埋め込み計算を回避し、後で差し替える
            clf = CustomLabelsClassifier(cls_ary=[species[0]], model_str=MODEL_STR, device=device)
            clf.classes = cache["classes"]
            clf.txt_embeddings = cache["emb"].to(device)
            return clf
        print("候補リストが変わったためキャッシュを再生成します。")

    # 初回（または候補変更時）: 全種の埋め込みを計算して保存
    print(f"テキスト埋め込みを計算中（{len(species)} 種、初回のみ・数分かかります）...")
    clf = CustomLabelsClassifier(cls_ary=species, model_str=MODEL_STR, device=device)
    torch.save({"key": key, "classes": clf.classes, "emb": clf.txt_embeddings.cpu()}, CACHE_PATH)
    print(f"キャッシュ保存: {CACHE_PATH}")
    return clf


def main():
    if len(sys.argv) < 2:
        print("使い方: python main.py <画像ファイルのパス> [<画像> ...]")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"使用デバイス: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("使用デバイス: cpu (GPUが利用できません)")

    species = load_candidates(CANDIDATE_CSV)
    print(f"候補種数: {len(species)}（{CANDIDATE_CSV}）")

    clf = build_classifier(species, device)

    for path in sys.argv[1:]:
        preds = clf.predict(path, k=TOP_K)
        print(f"\n画像: {path}")
        print("-" * 60)
        for p in preds:
            print(f"{p['classification']:<45} {p['score'] * 100:6.2f}%")


if __name__ == "__main__":
    main()
