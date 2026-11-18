# 021 — run commands (server)

Recipe fixed in `train.ipynb` Params: batch 16 × 512 px, 2,500 steps, lr 1e-3, log every 50 steps, checkpoint
every 500 steps. Only the architecture, loss and seed change between runs.

## tmux

```
tmux new -s train            # new session (first time)
tmux attach -t train         # back into it later
tmux ls                      # list sessions
```

Inside tmux: `Ctrl-b c` new window, `Ctrl-b n` / `Ctrl-b p` next / previous window, `Ctrl-b d` detach (training
keeps running). Never `Ctrl-c` in a training window unless you mean to stop it.

## Every window starts with

```
conda activate uavloc
cd /media/geekscode/Developer1/Projects/uavloc/src
EXP=experiments/021-building-loss-from-scratch
```

## One window per architecture (both seeds, all three losses)

```
sh $EXP/run_all.sh 42 unet-resnet34 && sh $EXP/run_all.sh 7 unet-resnet34
sh $EXP/run_all.sh 42 unet-resnet18 && sh $EXP/run_all.sh 7 unet-resnet18
sh $EXP/run_all.sh 42 lraspp-mobilenetv3 && sh $EXP/run_all.sh 7 lraspp-mobilenetv3
```

Start the R34 window first. Before opening the next window, check `nvidia-smi`: about 12 GB free for R18, about
6 GB for LR-ASPP.

## A single run (to redo one that failed)

```
papermill $EXP/train.ipynb $EXP/artifacts/run-<ARCH>-<LOSS>-seed<SEED>.ipynb -k uavloc --cwd $EXP --log-output \
  -p ARCHITECTURE <ARCH> -p LOSS <LOSS> -p SEED <SEED> -p NOTE "<ARCH>, <LOSS> loss, seed <SEED>"
```

| # | ARCH | LOSS | SEED |
|---|---|---|---|
| 1 | unet-resnet18 | current | 42 |
| 2 | unet-resnet18 | building | 42 |
| 3 | unet-resnet18 | structure | 42 |
| 4 | unet-resnet34 | current | 42 |
| 5 | unet-resnet34 | building | 42 |
| 6 | unet-resnet34 | structure | 42 |
| 7 | lraspp-mobilenetv3 | current | 42 |
| 8 | lraspp-mobilenetv3 | building | 42 |
| 9 | lraspp-mobilenetv3 | structure | 42 |
| 10–18 | same nine | | 7 |

Example, run 6:

```
papermill $EXP/train.ipynb $EXP/artifacts/run-unet-resnet34-structure-seed42.ipynb -k uavloc --cwd $EXP --log-output \
  -p ARCHITECTURE unet-resnet34 -p LOSS structure -p SEED 42 -p NOTE "unet-resnet34, structure loss, seed 42"
```

A failed run is already in `runs.jsonl` as `failed`; the rerun adds a new line, it does not replace it.

## Watch

```
nvidia-smi -l 5                                          # GPU memory and use
tensorboard --logdir tensorboard --port 6006             # in its own tmux window
ssh -N -L 6006:localhost:6006 geekscode@192.168.2.216    # on the Mac, then http://localhost:6006
wc -l $EXP/runs.jsonl                                    # finished runs (2 pilot + registered)
```
