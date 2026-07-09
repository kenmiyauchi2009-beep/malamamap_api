"""BioCLIP 2.5 Huge を CustomLabelsClassifier で試す（Myrtaceae 候補で前回と比較）。"""
import json
import time

from huggingface_hub import hf_hub_download
from bioclip import CustomLabelsClassifier

MODEL_25 = "hf-hub:imageomics/bioclip-2.5-vith14"
TARGET_FAMILY = "Myrtaceae"
FAMILY_IDX = 4  # scientific_names: [kingdom, phylum, class, order, family, genus, species_epithet]
IMAGE = "images.jpg"

# --- キャッシュ済みの種名JSONから Myrtaceae の二名法リストを作る（モデル不要） ---
names_path = hf_hub_download(
    repo_id="imageomics/TreeOfLife-200M",
    filename="embeddings/txt_emb_species.json",
    repo_type="dataset",
)
with open(names_path, encoding="utf-8") as fd:
    txt_names = json.load(fd)

species = []
for entry in txt_names:
    sci = entry[0]  # 学名のリスト
    if len(sci) >= FAMILY_IDX + 1 and sci[FAMILY_IDX] == TARGET_FAMILY:
        binomial = f"{sci[-2]} {sci[-1]}".strip()
        if binomial:
            species.append(binomial)
species = sorted(set(species))
print(f"候補種数（{TARGET_FAMILY}）: {len(species)}")

# --- BioCLIP 2.5 Huge で分類 ---
print("BioCLIP 2.5 Huge をロード中...")
t0 = time.time()
clf = CustomLabelsClassifier(cls_ary=species, model_str=MODEL_25, device="cuda")
print(f"テキスト埋め込み計算込みロード: {round(time.time() - t0, 1)} 秒")

preds = clf.predict(IMAGE, k=5)
print(f"\n[BioCLIP 2.5 Huge] 画像: {IMAGE}")
print("-" * 60)
for p in preds:
    print(f"{p['classification']:<45} {p['score'] * 100:6.2f}%")
