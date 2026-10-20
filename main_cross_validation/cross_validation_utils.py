"""
[File] cross_validation_utils.py
[Purpose] Utilities for Cross-Validation Data Management.
[Description]
    - Implements Stratified Group K-Fold splitting.
    - Implements Repeated Random Sub-sampling (GroupShuffleSplit) if valid_ratio is set.
    - Manages temporary dataset creation to prevent contamination.
    - Parses Scene IDs to ensure geographical isolation in splits.
[Author] Assistant (GitHub Copilot)
"""
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
import logging
import cv2

def parse_scene_id(filename):
    """
    從檔名解析場景 ID，用於避免 Data Leakage。
    假設格式: S1A_..._x1024_y512.png
    邏輯: 取最後一個 "_x" 之前的所有字串作為 ID。
    """
    filename = str(filename)
    if "_x" in filename:
        return filename.rsplit("_x", 1)[0]
    return filename

def parse_event_name(filepath: Path, base_dir: Path) -> str:
    """
    從路徑取得事件資料夾名稱（base_dir 下第一層子資料夾）。
    若檔案直接在 base_dir 底下，沒有事件子資料夾，回傳空字串。
    """
    try:
        rel = filepath.relative_to(base_dir)
        if len(rel.parts) > 1:
            return rel.parts[0]
    except ValueError:
        pass
    return ""

def count_scenes_per_event(dataset_path: Path, split_names) -> dict:
    """
    掃描指定 splits，統計每個事件資料夾下有多少個 unique SAR scene。
    用於決定該事件是按 event 還是按 scene 分組。
    """
    from collections import defaultdict
    event_scenes = defaultdict(set)
    for split_name in split_names:
        img_dir = dataset_path / 'images' / split_name
        if not img_dir.exists(): continue
        for f in img_dir.rglob('*'):
            if f.suffix.lower() not in ['.png', '.jpg', '.jpeg']: continue
            event_name = parse_event_name(f, img_dir)
            if not event_name: continue
            event_scenes[event_name].add(parse_scene_id(f.name))
    return {ev: len(scenes) for ev, scenes in event_scenes.items()}


def prepare_5fold_nested_split(source_patch_dir: Path,
                                k_outer: int = 5,
                                fix_last_fold_as_orig_test: bool = True,
                                scene_level_events: list = None,
                                random_state: int = 42) -> list:
    """
    [Nested CV] 把 source 內的 train+val+test patches 重新切成 K outer folds。

    Args:
        source_patch_dir: 原始 patch 資料夾，內含 images/{train,val,test}_{pos,neg}/...
        k_outer: outer fold 數量 (預設 5)
        fix_last_fold_as_orig_test: True → fold_{K-1} = 原本的 test (保留 baseline 可比性)
                                    False → 全部資料隨機重切
        scene_level_events: 列表中的事件按 scene 拆，其他按 event 綁定 (跟 prepare_kfold_datasets 同邏輯)
        random_state: 隨機種子

    Returns:
        folds: list of dict，長度 K_outer
          每個 dict 包含:
            'fold_idx': int
            'test_files': list[Path] - 該 fold 當 test 時用的檔案（全部，不 HNM filter）
            'test_is_pos': list[bool]
            'train_files': list[Path] - 其他 fold 合併後當 train (這些會餵給 HNM)
            'train_is_pos': list[bool]
            'description': str (audit 用)
    """
    import logging
    from sklearn.model_selection import StratifiedGroupKFold
    scene_level_events = set(scene_level_events or [])
    source_patch_dir = Path(source_patch_dir).resolve()

    # 1. 收集全部資料 (train + val + test) 統一處理
    records = []   # list of dict (path, is_pos, scene_id, event_name, orig_split)
    for orig_split in ['train', 'val', 'test']:
        for cat in ['pos', 'neg']:
            cat_dir = source_patch_dir / 'images' / f'{orig_split}_{cat}'
            if not cat_dir.exists(): continue
            for f in cat_dir.rglob('*'):
                if f.suffix.lower() not in ['.png', '.jpg', '.jpeg']: continue
                event_name = parse_event_name(f, cat_dir)
                scene_id   = parse_scene_id(f.name)
                records.append({
                    'path': f, 'is_pos': (cat == 'pos'),
                    'scene_id': scene_id, 'event_name': event_name,
                    'orig_split': orig_split,
                })
    if not records:
        raise ValueError(f"No files found under {source_patch_dir}/images/")

    df = pd.DataFrame(records)
    print(f"[5fold] 全部 patch 數 = {len(df)} (pos={df['is_pos'].sum()}, neg={(~df['is_pos']).sum()})")
    print(f"[5fold] 原 split 分佈: {df['orig_split'].value_counts().to_dict()}")

    # 2. 計算 group key (跟 prepare_kfold_datasets 同邏輯)
    def _group_key(row):
        if row['event_name'] and row['event_name'] not in scene_level_events:
            return f"EVENT::{row['event_name']}"
        return f"SCENE::{row['scene_id']}"
    df['group'] = df.apply(_group_key, axis=1)

    folds = [None] * k_outer
    used_indices = set()

    # 3. fold_{K-1} = 原本 test (如果開關)
    if fix_last_fold_as_orig_test:
        last_idx = k_outer - 1
        test_mask = df['orig_split'] == 'test'
        test_indices = df[test_mask].index.tolist()
        train_indices = df[~test_mask].index.tolist()
        folds[last_idx] = {
            'fold_idx': last_idx,
            'test_files': df.loc[test_indices, 'path'].tolist(),
            'test_is_pos': df.loc[test_indices, 'is_pos'].tolist(),
            'train_files': df.loc[train_indices, 'path'].tolist(),
            'train_is_pos': df.loc[train_indices, 'is_pos'].tolist(),
            'description': f'fold_{last_idx} = original test set (fixed)',
        }
        used_indices.update(test_indices)
        df_remain = df.drop(test_indices).reset_index(drop=True)
        n_remaining_folds = k_outer - 1
    else:
        df_remain = df.copy().reset_index(drop=True)
        n_remaining_folds = k_outer

    # 4. 對剩下的資料做 StratifiedGroupKFold，分到 fold_0..fold_{K-2}
    sgkf = StratifiedGroupKFold(n_splits=n_remaining_folds, shuffle=True, random_state=random_state)
    X = np.arange(len(df_remain))
    y = df_remain['is_pos'].astype(int).values
    groups = df_remain['group'].values

    fold_test_indices = {}   # fold_idx → list of indices in df_remain
    for f_idx_local, (_, test_idx_arr) in enumerate(sgkf.split(X, y, groups)):
        fold_test_indices[f_idx_local] = test_idx_arr.tolist()

    for f_idx_local, test_idx_arr in fold_test_indices.items():
        test_paths = df_remain.loc[test_idx_arr, 'path'].tolist()
        test_pos   = df_remain.loc[test_idx_arr, 'is_pos'].tolist()
        train_idx_arr = [i for i in range(len(df_remain)) if i not in set(test_idx_arr)]
        train_paths = df_remain.loc[train_idx_arr, 'path'].tolist()
        train_pos   = df_remain.loc[train_idx_arr, 'is_pos'].tolist()

        # 加上 fixed fold (orig test) 的 train portion
        if fix_last_fold_as_orig_test:
            train_paths = train_paths + folds[k_outer - 1]['test_files']   # ← 原 test 也成為這個 fold 的 train
            train_pos   = train_pos + folds[k_outer - 1]['test_is_pos']

        folds[f_idx_local] = {
            'fold_idx': f_idx_local,
            'test_files': test_paths,
            'test_is_pos': test_pos,
            'train_files': train_paths,
            'train_is_pos': train_pos,
            'description': f'fold_{f_idx_local} = StratifiedGroupKFold split #{f_idx_local}',
        }

    # 5. 列印分佈
    print(f"[5fold] === Outer fold 分佈統計 ===")
    for f in folds:
        n_test_pos = sum(f['test_is_pos'])
        n_test_neg = len(f['test_is_pos']) - n_test_pos
        n_tr_pos   = sum(f['train_is_pos'])
        n_tr_neg   = len(f['train_is_pos']) - n_tr_pos
        print(f"  fold_{f['fold_idx']}: "
              f"TRAIN(pos={n_tr_pos}, neg={n_tr_neg}) | "
              f"TEST(pos={n_test_pos}, neg={n_test_neg})  "
              f"# {f['description']}")

    return folds


def materialize_fold_as_symlinks(fold: dict,
                                  target_source_dir: Path,
                                  source_patch_dir: Path):
    """
    把單一 fold 的檔案結構物化成 SOURCE_PATCH_DIR 樣子的 symlink 樹，供 HNM script 讀取。

    target_source_dir/
       images/
          train_pos/    ← 該 fold 的 train pos symlinks
          train_neg/    ← 該 fold 的 train neg symlinks
          test_pos/     ← 該 fold 的 test pos symlinks
          test_neg/     ← 該 fold 的 test neg symlinks
       labels/...    (mirror 結構)

    為了不破壞 HNM 的 event subfolder 結構，會 preserve 原始路徑下的事件資料夾名。
    """
    target_source_dir = Path(target_source_dir)
    if target_source_dir.exists():
        shutil.rmtree(target_source_dir)

    def _link_pair(img_path, target_split_cat):
        """
        將 img + label 連結到 target_source_dir / images / target_split_cat / event_folder / file
        """
        img_path = Path(img_path)
        # 找出對應的 label 路徑（鏡像）：把 images/ → labels/、檔名同名
        src_img_dir_parts = img_path.parts
        if 'images' not in src_img_dir_parts:
            return  # 結構不對，跳過
        idx = src_img_dir_parts.index('images')
        src_lbl_parts = list(src_img_dir_parts)
        src_lbl_parts[idx] = 'labels'

        # 對 PNG label
        for suf in ['.png', '.txt']:
            lbl_path = Path(*src_lbl_parts).with_suffix(suf)
            if lbl_path.exists(): break
        else:
            lbl_path = None

        # 計算事件子資料夾（在 split_cat 下的相對 parent）
        src_cat_dir = source_patch_dir / 'images' / src_img_dir_parts[idx + 1]
        try:
            rel_parent = img_path.parent.relative_to(src_cat_dir)
        except ValueError:
            rel_parent = Path('.')

        tgt_img_dir = target_source_dir / 'images' / target_split_cat / rel_parent
        tgt_img_dir.mkdir(parents=True, exist_ok=True)
        tgt_img = tgt_img_dir / img_path.name
        if not tgt_img.exists():
            os.symlink(img_path.resolve(), tgt_img)

        if lbl_path and lbl_path.exists():
            tgt_lbl_dir = target_source_dir / 'labels' / target_split_cat / rel_parent
            tgt_lbl_dir.mkdir(parents=True, exist_ok=True)
            tgt_lbl = tgt_lbl_dir / lbl_path.name
            if not tgt_lbl.exists():
                os.symlink(lbl_path.resolve(), tgt_lbl)

    # train 部分 → train_pos / train_neg
    for p, is_pos in zip(fold['train_files'], fold['train_is_pos']):
        _link_pair(p, 'train_pos' if is_pos else 'train_neg')
    # test 部分 → test_pos / test_neg
    for p, is_pos in zip(fold['test_files'], fold['test_is_pos']):
        _link_pair(p, 'test_pos' if is_pos else 'test_neg')

    # 為了 HNM 內部 K-fold 工作，需要 val_pos / val_neg。簡化：把 train 一小部分當 val（symlink）
    # 直接把 train 全 link 一份到 val (HNM 內部 StratifiedGroupKFold 會 split)
    # 實際上 HNM 是讀 train+val 合併處理，所以這裡讓 val 為空也行 (HNM 會自己分 fold)
    # 為與既有 HNM script 相容，建立空的 val 目錄
    for cat in ['val_pos', 'val_neg']:
        (target_source_dir / 'images' / cat).mkdir(parents=True, exist_ok=True)
        (target_source_dir / 'labels' / cat).mkdir(parents=True, exist_ok=True)


def _infer_is_pos_from_symlink(path: Path) -> bool:
    """從 symlink 最終目標路徑反推是否正樣本 (path 結構含 train_pos/train_neg/...)。"""
    try:
        target = Path(path).resolve()
    except Exception:
        return False
    for part in target.parts:
        if part.endswith('_pos'):
            return True
        if part.endswith('_neg'):
            return False
    return False  # 未知 → 保守當 neg


def materialize_train_val_split(fold_dataset_path: Path,
                                  scene_level_events: list = None,
                                  val_ratio: float = 0.1,
                                  seed: int = 42,
                                  manifest_xlsx: Path = None) -> Path:
    """從 fold 的 HNM dataset (只有 train/test) 切出 val，產出 sibling symlink tree。

    輸入結構 (HNM Phase 3 輸出，path 已含 fold/Global_Dataset_Ratio_*)：
        fold_dataset_path/
          images/train/<event>/<file>.png
          images/test/<event>/<file>.png
          labels/...

    輸出結構 (新建在 fold_dataset_path 旁邊)：
        fold_dataset_path__trainval/
          images/train/<event>/...    ← (1 - val_ratio) symlinks
          images/val/<event>/...      ← val_ratio symlinks (按 grouping 隨機抽)
          images/test/<event>/...     ← 全部 symlinks (對齊原 test)
          labels/...

    Grouping (跟 prepare_5fold_nested_split 同邏輯)：
        - event 在 scene_level_events 白名單 → 按 scene_id 分組
        - 其他 event → 整 event 綁同邊 (train 或 val)
        - 用 GroupShuffleSplit (single split) 切 train / val

    manifest_xlsx 若給定，會寫一份切分稽核檔（Summary / Files / GroupBreakdown 三 sheet）。

    Idempotent: 若 target dir 已存在且 images/val/ 非空，跳過重做 (回傳路徑，仍會嘗試寫 manifest)。
    """
    from sklearn.model_selection import GroupShuffleSplit
    import numpy as np

    fold_dataset_path = Path(fold_dataset_path).resolve()
    target_dir = fold_dataset_path.parent / (fold_dataset_path.name + "__trainval")
    scene_level_events = set(scene_level_events or [])

    src_train_imgs = fold_dataset_path / "images" / "train"
    src_train_lbls = fold_dataset_path / "labels" / "train"
    src_test_imgs  = fold_dataset_path / "images" / "test"
    src_test_lbls  = fold_dataset_path / "labels" / "test"
    if not src_train_imgs.exists():
        raise RuntimeError(f"[train_val_split] 找不到 {src_train_imgs}")

    # 收集 train files + 算 group key
    train_records = []  # (img_path, event_name, scene_id, group_key, is_pos)
    for img_path in src_train_imgs.rglob("*.png"):
        event_name = parse_event_name(img_path, src_train_imgs)
        scene_id   = parse_scene_id(img_path.name)
        if event_name and event_name not in scene_level_events:
            group_key = f"EVENT::{event_name}"
        else:
            group_key = f"SCENE::{scene_id}"
        is_pos = _infer_is_pos_from_symlink(img_path)
        train_records.append((img_path, event_name, scene_id, group_key, is_pos))

    if not train_records:
        raise RuntimeError(f"[train_val_split] {src_train_imgs} 沒找到任何 .png")

    n_total = len(train_records)
    groups  = np.array([r[3] for r in train_records])
    n_groups = len(set(groups))

    # Cache invalidation：用 marker file 記錄上次切分的 config
    # 同樣 config 就 skip rebuild；config 改了 (val_ratio / seed / scene_level_events) 自動 rebuild
    marker_file = target_dir / "_split_config.txt"
    expected_marker = (
        f"val_ratio={val_ratio}\n"
        f"seed={seed}\n"
        f"scene_level_events={sorted(scene_level_events)}\n"
    )
    val_img_dir = target_dir / "images" / "val"
    skip_rebuild = (val_img_dir.exists()
                     and any(val_img_dir.rglob("*.png"))
                     and marker_file.exists()
                     and marker_file.read_text() == expected_marker)

    if skip_rebuild:
        print(f"[train_val_split] 已存在 {target_dir.name} 且 config 一致，跳過 symlink 重建")
        # 還原 val_set 從現有 val_img_dir，給 manifest 寫入用
        val_set = set()
        for vp in val_img_dir.rglob("*.png"):
            try:
                rel = vp.relative_to(val_img_dir)
            except ValueError:
                continue
            cand = src_train_imgs / rel
            if cand.exists() or cand.is_symlink():
                val_set.add(cand)
    else:
        if target_dir.exists() and marker_file.exists():
            old = marker_file.read_text()
            print(f"[train_val_split] config 已變更，重建 {target_dir.name}\n"
                  f"  舊: {old.strip().replace(chr(10), ' / ')}\n"
                  f"  新: {expected_marker.strip().replace(chr(10), ' / ')}")
        # 用 GroupShuffleSplit (1 split) 切 train / val
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(gss.split(X=np.arange(n_total), groups=groups))
        val_set = {train_records[i][0] for i in val_idx}

        # 清空 target dir 重建
        if target_dir.exists():
            shutil.rmtree(target_dir)

        def _mirror_link(src_img: Path, src_imgs_root: Path, src_lbls_root: Path,
                          tgt_split: str):
            try:
                rel = src_img.relative_to(src_imgs_root)
            except ValueError:
                rel = Path(src_img.name)
            tgt_img = target_dir / "images" / tgt_split / rel
            tgt_img.parent.mkdir(parents=True, exist_ok=True)
            if not tgt_img.exists():
                os.symlink(src_img.resolve(), tgt_img)
            for suf in ['.png', '.txt']:
                src_lbl = (src_lbls_root / rel).with_suffix(suf)
                if src_lbl.exists():
                    tgt_lbl = (target_dir / "labels" / tgt_split / rel).with_suffix(suf)
                    tgt_lbl.parent.mkdir(parents=True, exist_ok=True)
                    if not tgt_lbl.exists():
                        os.symlink(src_lbl.resolve(), tgt_lbl)

        for img_path, _, _, _, _ in train_records:
            tgt_split = 'val' if img_path in val_set else 'train'
            _mirror_link(img_path, src_train_imgs, src_train_lbls, tgt_split)

        if src_test_imgs.exists():
            for img_path in src_test_imgs.rglob("*.png"):
                _mirror_link(img_path, src_test_imgs, src_test_lbls, 'test')

        # 寫 marker file 記錄這次切分的 config (下次同 config 才會 skip)
        marker_file.write_text(expected_marker)

        print(f"[train_val_split] {fold_dataset_path.name}: "
              f"total={n_total}, n_groups={n_groups}, val_ratio={val_ratio}, "
              f"→ train={n_total - len(val_set)}, val={len(val_set)}")
        print(f"[train_val_split] 輸出 → {target_dir}")

    # 寫 manifest xlsx (no matter rebuild 或 skip 都會寫一份，方便每 fold 各自存到 result dir)
    if manifest_xlsx is not None:
        _write_train_val_manifest(
            manifest_xlsx=Path(manifest_xlsx),
            fold_dataset_path=fold_dataset_path,
            train_records=train_records,
            val_set=val_set,
            src_test_imgs=src_test_imgs,
            scene_level_events=scene_level_events,
            val_ratio=val_ratio,
            seed=seed,
        )

    return target_dir


def _write_train_val_manifest(manifest_xlsx: Path,
                                fold_dataset_path: Path,
                                train_records: list,
                                val_set: set,
                                src_test_imgs: Path,
                                scene_level_events: set,
                                val_ratio: float,
                                seed: int):
    """寫 train/val/test 切分稽核 excel (Summary / Files / GroupBreakdown)。"""
    import pandas as pd

    file_rows = []
    # train + val
    for img_path, event, scene, group, is_pos in train_records:
        role = 'val' if img_path in val_set else 'train'
        file_rows.append({
            'role': role,
            'event_name': event,
            'scene_id': scene,
            'group_key': group,
            'group_strategy': 'scene' if group.startswith('SCENE::') else 'event',
            'is_pos': bool(is_pos),
            'path': str(img_path),
        })
    # test
    if src_test_imgs.exists():
        for img_path in src_test_imgs.rglob("*.png"):
            event = parse_event_name(img_path, src_test_imgs)
            scene = parse_scene_id(img_path.name)
            is_pos = _infer_is_pos_from_symlink(img_path)
            file_rows.append({
                'role': 'test',
                'event_name': event,
                'scene_id': scene,
                'group_key': '',
                'group_strategy': '',
                'is_pos': bool(is_pos),
                'path': str(img_path),
            })

    df_files = pd.DataFrame(file_rows)

    # Summary: per role 計數
    summary_rows = []
    for role in ['train', 'val', 'test']:
        sub = df_files[df_files['role'] == role]
        summary_rows.append({
            'role': role,
            'n_total': len(sub),
            'n_pos':   int(sub['is_pos'].sum()) if len(sub) else 0,
            'n_neg':   int((~sub['is_pos'].astype(bool)).sum()) if len(sub) else 0,
        })
    summary_rows.append({
        'role': 'config',
        'n_total': f"val_ratio={val_ratio}, seed={seed}, "
                    f"scene_level_events={sorted(scene_level_events)}",
        'n_pos': '',
        'n_neg': '',
    })
    summary_rows.append({
        'role': 'source',
        'n_total': str(fold_dataset_path),
        'n_pos': '', 'n_neg': '',
    })
    df_summary = pd.DataFrame(summary_rows)

    # GroupBreakdown: per (role, event, group_strategy) 計數
    df_trainval = df_files[df_files['role'].isin(['train', 'val'])]
    if not df_trainval.empty:
        grp = df_trainval.groupby(['role', 'event_name', 'group_strategy'], dropna=False)
        df_group = grp.agg(
            n_files=('path', 'count'),
            n_pos=('is_pos', 'sum'),
        ).reset_index()
        df_group['n_neg'] = df_group['n_files'] - df_group['n_pos']
        df_group = df_group[['role', 'event_name', 'group_strategy',
                              'n_files', 'n_pos', 'n_neg']]
    else:
        df_group = pd.DataFrame()

    manifest_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(manifest_xlsx) as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_group.to_excel(writer, sheet_name='GroupBreakdown', index=False)
        df_files.to_excel(writer, sheet_name='Files', index=False)
    print(f"[train_val_split] manifest 寫入: {manifest_xlsx}")


def get_sample_class(label_path):
    """
    讀取標籤內容以判斷是正樣本(1)還是負樣本(0)。
    """
    if not label_path.exists():
        return 0
    
    # 針對 YOLO .txt 格式
    if label_path.suffix == '.txt':
        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
            return 1 if len(lines) > 0 else 0
            
    # 針對 Mask .png 格式
    elif label_path.suffix in ['.png', '.jpg']:
        try:
            mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if mask is None: return 0
            return 1 if cv2.countNonZero(mask) > 0 else 0
        except:
            return 0
    return 0

def link_files(indices, source_imgs, source_lbls_list, target_img_dir, target_lbl_dir):
    """建立符號連結 (Symlink) 以節省空間。
    支援事件子資料夾：若圖片在子資料夾中（parent.name != split dir name），
    以 '{事件名}___{檔名}' 展平命名，避免不同事件間的同名衝突。
    """
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in indices:
        src_img = source_imgs[i]

        # 若圖片的父資料夾名稱與 target_img_dir 的 split 名不同，表示有事件子資料夾，加前綴
        if src_img.parent.name != target_img_dir.name:
            flat_stem = f"{src_img.parent.name}___{src_img.stem}"
        else:
            flat_stem = src_img.stem

        dst_img = target_img_dir / (flat_stem + src_img.suffix)
        if not dst_img.exists():
            os.symlink(src_img.resolve(), dst_img)

        labels = source_lbls_list[i]
        for src_lbl in labels:
            if src_lbl.exists():
                dst_lbl = target_lbl_dir / (flat_stem + src_lbl.suffix)
                if not dst_lbl.exists():
                    os.symlink(src_lbl.resolve(), dst_lbl)

def scan_dataset_split(dataset_path, split_name, scene_level_events=None):
    """
    掃描特定 split 並回傳檔案資訊與分組鍵。

    分組策略：
      - 預設：按 event（事件資料夾名）綁定 → grp = "EVENT::事件名"
              整個事件的所有 SAR scene 進同一個 fold，避免地點洩漏
      - 例外：event 在 scene_level_events 白名單內 → 按 scene 拆 → grp = "SCENE::xxx"
              讓大事件的不同日期可以分散到多個 fold，避免單一 fold 樣本失衡
      - 邊界：無事件資料夾的檔案 → 自動退化為 scene-level
    """
    img_dir = dataset_path / 'images' / split_name
    lbl_dir = dataset_path / 'labels' / split_name

    imgs, lbls_list, grps, clss = [], [], [], []

    if not img_dir.exists():
        return imgs, lbls_list, grps, clss

    scene_level_events = set(scene_level_events or [])
    files = sorted([f for f in img_dir.rglob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])

    for f in files:
        rel_parent = f.parent.relative_to(img_dir)

        found_labels = []
        l_txt = lbl_dir / rel_parent / f"{f.stem}.txt"
        l_png = lbl_dir / rel_parent / f"{f.stem}.png"
        if l_txt.exists(): found_labels.append(l_txt)
        if l_png.exists(): found_labels.append(l_png)

        cls = 0
        target_lbl_for_class = l_png if l_png.exists() else (l_txt if l_txt.exists() else None)
        if target_lbl_for_class:
            cls = get_sample_class(target_lbl_for_class)

        event_name = parse_event_name(f, img_dir)
        scene_id   = parse_scene_id(f.name)
        if event_name and event_name not in scene_level_events:
            grp = f"EVENT::{event_name}"
        else:
            grp = f"SCENE::{scene_id}"

        imgs.append(f)
        lbls_list.append(found_labels)
        grps.append(grp)
        clss.append(cls)

    return imgs, lbls_list, grps, clss

def prepare_kfold_datasets(original_dataset_path, results_base_dir, k_folds=3, seed=42, cv_settings=None):
    """
    執行 Stratified Group K-Fold 或 Group Shuffle Split 拆分，建立暫存資料夾。
    """
    if cv_settings is None: cv_settings = {}
    
    # [關鍵修正] 使用 .resolve() 強制轉換為絕對路徑
    original_path = Path(original_dataset_path).expanduser().resolve()
    
    cv_work_dir = Path(results_base_dir) / "cv_temp" / original_path.name
    
    # 若資料夾已存在，先移除以確保乾淨
    if cv_work_dir.exists():
        print(f"[CV] 清理舊的 CV 暫存資料夾: {cv_work_dir}")
        shutil.rmtree(cv_work_dir)
    
    # 0. [Group strategy] 預設按 event 綁定（整事件進同一 fold，避免地點洩漏）；
    #    user 可在 cv_settings.scene_level_events 列出要按 scene 拆的事件白名單。
    scene_level_events = set(cv_settings.get('scene_level_events', []))
    event_scene_count  = count_scenes_per_event(original_path, ['train', 'val'])

    print(f"[CV] Group strategy = default EVENT-bound, "
                 f"scene-level whitelist size = {len(scene_level_events)}")

    # 完整事件清單（按 scene 數降序）與分組標示
    sorted_events = sorted(event_scene_count.items(), key=lambda x: -x[1])
    for ev, cnt in sorted_events:
        marker = "[SCENE]" if ev in scene_level_events else "[EVENT]"
        print(f"     {marker} {ev:40s} : {cnt:4d} unique scenes")

    # 警告白名單中找不到的事件名（可能拼錯）
    missing = scene_level_events - set(event_scene_count.keys())
    if missing:
        logging.warning(f"[CV] scene_level_events 中以下事件在資料集找不到（請檢查拼字）: {missing}")

    # 1. 掃描 Train 與 Val (合併為開發集)
    print(f"[CV] 正在掃描 Train/Val 資料集 (來源: {original_path})...")
    tr_imgs, tr_lbls, tr_grps, tr_clss = scan_dataset_split(
        original_path, 'train', scene_level_events)
    val_imgs, val_lbls, val_grps, val_clss = scan_dataset_split(
        original_path, 'val', scene_level_events)
    
    # 合併
    all_imgs = tr_imgs + val_imgs
    all_lbls = tr_lbls + val_lbls #這現在是 list of lists
    all_groups = tr_grps + val_grps
    all_classes = tr_clss + val_clss

    X = np.array(all_imgs)
    y = np.array(all_classes)
    groups = np.array(all_groups)
    
    # 1.1 額外掃描 Test Set (不參與拆分，但需要統計)
    print("[CV] 正在掃描 Test 資料集以進行統計...")
    test_imgs, _, _, test_clss = scan_dataset_split(original_path, 'test')
    test_pos_count = sum(test_clss)
    test_neg_count = len(test_clss) - test_pos_count
    
    # 準備 Excel 紀錄 DataFrame (詳細清單)
    record_df = pd.DataFrame({
        'filename': [p.name for p in X],
        'scene_id': groups,
        'class': y, # 1=Pos, 0=Neg
        'original_path': [str(p) for p in X]
    })

    # 準備 統計摘要 列表
    summary_stats = []

    # 2. 決定拆分策略
    val_ratio = cv_settings.get('val_ratio', None)
    
    if val_ratio is not None:
        print(f"[CV] 使用 Repeated Group Shuffle Split (Train={1-val_ratio:.0%}, Val={val_ratio:.0%})")
        # 注意: GroupShuffleSplit 不保證 Stratified (類別平衡)，但保證 Group 不洩漏 (地理位置隔離)
        # 在 M4D 這類地理資料集中，Group 隔離通常比 Stratified 更重要來避免 Data Leakage
        splitter = GroupShuffleSplit(n_splits=k_folds, test_size=val_ratio, random_state=seed)
        split_gen = splitter.split(X, y, groups)
    else:
        print(f"[CV] 使用標準 Stratified Group K-Fold (k={k_folds})")
        splitter = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        split_gen = splitter.split(X, y, groups)
    
    fold_paths = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(split_gen):
        fold_name = f"fold_{fold_idx}" # 0-based
        fold_root = cv_work_dir / fold_name
        
        # 更新 Excel 詳細紀錄
        record_df[f'Fold_{fold_idx+1}_Role'] = '' # 1-based for Excel
        record_df.loc[train_idx, f'Fold_{fold_idx+1}_Role'] = 'Train'
        record_df.loc[val_idx, f'Fold_{fold_idx+1}_Role'] = 'Val'
        
        # 計算統計數據
        tr_pos = sum(y[train_idx])
        tr_neg = len(train_idx) - tr_pos
        val_pos = sum(y[val_idx])
        val_neg = len(val_idx) - val_pos
        
        print(f"[CV] {fold_name}: "
                     f"Train(Pos:{tr_pos}, Neg:{tr_neg}) | "
                     f"Val(Pos:{val_pos}, Neg:{val_neg})")
        
        summary_stats.append({
            'Fold': fold_idx + 1,
            'Train_Pos': tr_pos, 'Train_Neg': tr_neg, 'Train_Total': len(train_idx),
            'Val_Pos': val_pos, 'Val_Neg': val_neg, 'Val_Total': len(val_idx),
            'Test_Pos': test_pos_count, 'Test_Neg': test_neg_count, 'Test_Total': len(test_clss)
        })
        
        # 3. 建立連結 (Train/Val)
        link_files(train_idx, X, all_lbls, fold_root / 'images' / 'train', fold_root / 'labels' / 'train')
        link_files(val_idx, X, all_lbls, fold_root / 'images' / 'val', fold_root / 'labels' / 'val')
        
        # 4. 處理 Test (Symlink 原始 Test)
        orig_test_img = original_path / 'images' / 'test'
        orig_test_lbl = original_path / 'labels' / 'test'
        if orig_test_img.exists():
            target_t_img = fold_root / 'images' / 'test'
            if not target_t_img.exists():
                 target_t_img.symlink_to(orig_test_img.resolve())
        if orig_test_lbl.exists():
            target_t_lbl = fold_root / 'labels' / 'test'
            if not target_t_lbl.exists():
                target_t_lbl.symlink_to(orig_test_lbl.resolve())
            
        fold_paths.append(str(fold_root))
    
    # 儲存 Excel (兩個 Sheet)
    excel_path = cv_work_dir / f"{original_path.name}_CV_Split_Info.xlsx"
    cv_work_dir.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(excel_path) as writer:
        pd.DataFrame(summary_stats).to_excel(writer, sheet_name='Summary_Stats', index=False)
        record_df.to_excel(writer, sheet_name='Split_Details', index=False)
        
    print(f"[CV] 分割詳細資訊與統計已儲存至: {excel_path}")
        
    return fold_paths