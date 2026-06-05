#!/usr/bin/env python3
"""
Write 71-class SCP-ECG labels from ptbxl_database.csv into each HR*.mat file.
Run once before classification finetuning.

Usage:
    python src/preprocess/add_labels_to_mat.py
    python src/preprocess/add_labels_to_mat.py --mat-dir /path/to/mat --csv /path/to/ptbxl_database.csv
"""
import argparse
import os
import sys
import scipy.io
import numpy as np
import pandas as pd

CLASS_LABELS = [
    '1AVB','2AVB','3AVB','ABQRS','AFIB','AFLT','ALMI','AMI','ANEUR','ASMI',
    'BIGU','CLBBB','CRBBB','DIG','EL','HVOLT','ILBBB','ILMI','IMI','INJAL',
    'INJAS','INJIL','INJIN','INJLA','INVT','IPLMI','IPMI','IRBBB','ISCAL',
    'ISCAN','ISCAS','ISCIL','ISCIN','ISCLA','ISC_','IVCD','LAFB','LAO/LAE',
    'LMI','LNGQT','LOWT','LPFB','LPR','LVH','LVOLT','NDT','NORM','NST_',
    'NT_','PAC','PACE','PMI','PRC(S)','PSVT','PVC','QWAVE','RAO/RAE','RVH',
    'SARRH','SBRAD','SEHYP','SR','STACH','STD_','STE_','SVARR','SVTAC',
    'TAB_','TRIGU','VCLVH','WPW',
]
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(CLASS_LABELS)}
NUM_LABELS = len(CLASS_LABELS)  # 71

MATLAB_META_KEYS = {'__header__', '__version__', '__globals__'}


def main():
    parser = argparse.ArgumentParser()
    proj_default = os.environ.get(
        'PROJ_DIR',
        os.path.join(os.path.dirname(__file__), '..', '..')
    )
    parser.add_argument('--mat-dir', default=os.path.join(proj_default, 'data', 'ptbxl_mat'))
    parser.add_argument('--csv', default=os.path.join(proj_default, 'src', 'ptbxl_files', 'ptbxl_database.csv'))
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip .mat files that already have a label key')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        alt = os.path.join(proj_default, 'ptbxl_files', 'ptbxl_database.csv')
        if os.path.exists(alt):
            args.csv = alt
        else:
            print(f"ERROR: ptbxl_database.csv not found at {args.csv}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading {args.csv} ...")
    db = pd.read_csv(args.csv)
    ecg_to_codes: dict = {}
    for _, row in db.iterrows():
        ecg_to_codes[int(row['ecg_id'])] = set(eval(row['scp_codes']).keys())
    print(f"  Loaded {len(ecg_to_codes)} ECG records from database.")

    mat_files = sorted(f for f in os.listdir(args.mat_dir) if f.endswith('.mat'))
    total = len(mat_files)
    print(f"Processing {total} .mat files in {args.mat_dir} ...")

    updated = skipped = missing = errors = 0
    for i, fname in enumerate(mat_files):
        mat_path = os.path.join(args.mat_dir, fname)

        # fname format: HR{ecg_id}.mat
        try:
            ecg_id = int(fname[2:-4])
        except ValueError:
            errors += 1
            continue

        if args.skip_existing:
            try:
                probe = scipy.io.loadmat(mat_path, variable_names=['label'])
                if 'label' in probe:
                    skipped += 1
                    continue
            except Exception:
                pass

        if ecg_id not in ecg_to_codes:
            missing += 1
            continue

        label = np.zeros(NUM_LABELS, dtype=np.float32)
        for code in ecg_to_codes[ecg_id]:
            if code in LABEL_TO_IDX:
                label[LABEL_TO_IDX[code]] = 1.0

        try:
            data = scipy.io.loadmat(mat_path)
            for k in MATLAB_META_KEYS:
                data.pop(k, None)
            data['label'] = label.reshape(1, -1)  # [1, 71] — squeeze(0) → [71]
            scipy.io.savemat(mat_path, data)
            updated += 1
        except Exception as e:
            print(f"  ERROR on {fname}: {e}", file=sys.stderr)
            errors += 1

        if (i + 1) % 2000 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] updated={updated} skipped={skipped} missing={missing} errors={errors}")

    print(f"\nDone. updated={updated}, skipped={skipped}, missing={missing}, errors={errors}")


if __name__ == '__main__':
    main()
