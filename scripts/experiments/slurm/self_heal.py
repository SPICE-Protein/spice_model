#!/usr/bin/env python3
import sys

if sys.version_info[0] < 3:
    sys.exit("需要 Python 3：先 conda activate spice，或用 ~/miniconda3/envs/spice/bin/python")

"""幂等：把某尺度 posttrain.yaml 的输出字段指到该尺度自己的目录（防跨尺度互覆）。
用法： python self_heal.py <posttrain.yaml 路径> <scale N>
"""
import yaml

p, n = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open(p))
base = f"../data/data_efficiency/n{n}"
c["post"]["pseudo_label_dir"] = f"{base}/pseudo_labels"
c["post"]["pseudo_tfrecord_path"] = f"{base}/pseudo.tfrecord"
c["post"]["phase_map_dir"] = f"{base}/phase_maps"
yaml.safe_dump(c, open(p, "w"), allow_unicode=True)
print(f"[self-heal] n{n} 输出隔离 OK")
