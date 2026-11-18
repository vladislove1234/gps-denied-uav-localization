# 021 on the training server

Linux, RTX 3090 Ti (24 GB), driver CUDA 13.0, internet. All 18 runs happen there, so their numbers compare with
each other and not with anything trained on the Mac (different RNG streams).

## 1. Copy (from the Mac, project root)

The 021 code is uncommitted, so copy the working tree rather than cloning. Keep the layout: `data/` next to `src/`,
and `src/data` a symlink to `../data`.

```
SERVER=user@host; DEST=uav
rsync -av --exclude '.git/' --exclude '*.pt' --exclude '*.ts' --exclude '/tensorboard/' \
      --exclude 'experiments/*/artifacts/' --exclude '**/runs/' --exclude '.venv/' --exclude '__pycache__/' \
      --exclude '*.egg-info/' --exclude '/data' src/ $SERVER:$DEST/src/
ssh $SERVER "mkdir -p $DEST/data/derived/ua-village-multidate"
rsync -av data/derived/ua-village-annotated $SERVER:$DEST/data/derived/
rsync -av data/derived/ua-village-multidate/birky-49p9105-23p9192 \
      data/derived/ua-village-multidate/bushcha-50p2422-25p8305 \
      data/derived/ua-village-multidate/dubrova-49p5175-24p0765 $SERVER:$DEST/data/derived/ua-village-multidate/
ssh $SERVER "cd $DEST/src && ln -sfn ../data data"
```

About 31 MB of data plus the code. No `rsync -R` with `/./` markers: the Mac's rsync ignores them and
nests the whole path. Excludes of top-level paths start with `/`: a bare `data` would also skip
the `uavloc/data/` package. `.git/` is left out, so `log_run` records no commit; add it back (drop that
exclude) if you want the hash in `runs.jsonl`.

## 2. Environment (on the server, once)

```
conda create -n uavloc python=3.11 -y && conda activate uavloc
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
cd ~/uav/src && pip install -r requirements-server.txt && pip install -e .
python -m ipykernel install --user --name uavloc
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The cu128 wheel runs on the CUDA 13 driver. ImageNet weights download on first use.

## 3. Checks (on the server, before training)

```
cd ~/uav/src && EXP=experiments/021-building-loss-from-scratch && mkdir -p $EXP/artifacts
papermill $EXP/checks.ipynb $EXP/artifacts/checks.ipynb -k uavloc --cwd $EXP
```

Every assert must pass; open `artifacts/checks.ipynb` to see the numbers and the figure.

## 4. Train

In the registered order, one run after another:

```
tmux new -s train
cd ~/uav/src && conda activate uavloc
sh experiments/021-building-loss-from-scratch/run_all.sh 42 && sh experiments/021-building-loss-from-scratch/run_all.sh 7
```

Or in parallel, one architecture per tmux window (`Ctrl-b c` for a new window), if the first run shows the GPU is
mostly idle:

```
sh experiments/021-building-loss-from-scratch/run_all.sh 42 unet-resnet18 && sh experiments/021-building-loss-from-scratch/run_all.sh 7 unet-resnet18
sh experiments/021-building-loss-from-scratch/run_all.sh 42 unet-resnet34 && sh experiments/021-building-loss-from-scratch/run_all.sh 7 unet-resnet34
sh experiments/021-building-loss-from-scratch/run_all.sh 42 lraspp-mobilenetv3 && sh experiments/021-building-loss-from-scratch/run_all.sh 7 lraspp-mobilenetv3
```

Detach with `Ctrl-b d`, come back with `tmux attach -t train`. Each run's cell output streams into its window
(`--log-output`) and is kept in `artifacts/run-<arch>-<loss>-seed<N>.ipynb`.

## 5. Watch

```
tensorboard --logdir ~/uav/src/tensorboard --port 6006          # on the server
ssh -L 6006:localhost:6006 user@host                            # on the Mac, then open http://localhost:6006
nvidia-smi -l 5                                                 # GPU use and memory
```

## 6. Bring back (from the Mac, project root)

```
rsync -av $SERVER:$DEST/src/experiments/021-building-loss-from-scratch/{runs.jsonl,figures,artifacts} \
      src/experiments/021-building-loss-from-scratch/
rsync -av $SERVER:$DEST/src/tensorboard/021-building-loss-from-scratch src/tensorboard/
```

`artifacts/` holds the checkpoints (U-Net R34 about 98 MB each); add `--exclude '*.pt'` to leave them on the server.
