#!/usr/bin/env bash
# sonic 全量 iLQR: solved-only + 默认早退 + 参考相对摔倒判定 + 提前否决 + 行程归一化根门。
# 按帧数贪心均衡分片; 可随时 pkill 中断, 重跑跳过已 solved 的 clip。
# 用法: [SUBSET=正则] [LIMIT=N] [NS=32] [CORES=5] [KAPPA=0.10] ./launch_sonic_ilqr.sh
cd /home/unilab/jiaxi/Unilab_fork_offpolicy_sonic
OUT=data/sonic_ilqr; NS="${NS:-32}"; CORES="${CORES:-5}"; KAPPA="${KAPPA:-0.10}"
SUBSET="${SUBSET:-}"; LIMIT="${LIMIT:-0}"
mkdir -p $OUT shards_sonic_ilqr
.venv/bin/python - <<PY
import numpy as np
names = [l.strip() for l in open("data/sonic/packed/clip_names.txt") if l.strip()]
lens = np.load("data/sonic/packed/clip_lengths.npy")
NS = $NS
import re
subset = re.compile("$SUBSET") if "$SUBSET" else None
pairs = [(n, int(lens[i])) for i, n in enumerate(names)]
if subset:
    pairs = [(n, l) for n, l in pairs if subset.search(n)]
if $LIMIT > 0:
    pairs = pairs[:$LIMIT]
shards = [[] for _ in range(NS)]
loads = [0] * NS
for n, l in sorted(pairs, key=lambda p: -p[1]):
    j = loads.index(min(loads))
    shards[j].append(n); loads[j] += l
for j in range(NS):
    if shards[j]:
        open(f"shards_sonic_ilqr/s{j}.txt", "w").write("\n".join(shards[j]) + "\n")
print("balanced", NS, "shards over", len(pairs), "clips; max shard load", max(loads), "frames")
PY
for f in logs_sonic_ilqr_s*.log; do [ -f "$f" ] && mv "$f" "$f.run$(date +%m%d_%H%M)"; done
for ((i=0; i<NS; i++)); do
  s=$((i*CORES)); e=$(((i+1)*CORES-1))
  nohup taskset -c $s-$e .venv/bin/python scripts/motion/mocap2ilqr.py \
    --source data/sonic --output $OUT --jobs 1 \
    --clips-file shards_sonic_ilqr/s$i.txt \
    --root-path-kappa $KAPPA --skip-pack \
    > logs_sonic_ilqr_s$i.log 2>&1 &
done
echo "launched $NS shards x $CORES cores, kappa=$KAPPA -> $OUT"
