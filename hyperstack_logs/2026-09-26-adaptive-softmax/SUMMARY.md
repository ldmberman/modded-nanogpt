# Adaptive softmax runs on Hyperstack, 2026-09-26

All runs used one H100 spot VM (Hyperstack id 1066857, creative-rutherford). The four queued jobs (10, 20, 30, 40) ran between 22:00:59 and 23:10:09 UTC according to `history.log`. The two earlier runs, `run_adaptive.log` and `run_baseline.log`, were launched by hand with a random seed. Their command lines were not recorded in the queue, so the command column says so. Every number below was taken from the console logs with grep. The matching full training logs are saved next to them as `<run log name>.full.txt`.

Job 40 crashed during compile warmup with a CUDA out-of-memory error, before it logged any training step. The queue history still records it as `exit=0`, so the queue's exit codes do not reliably show crashes. Jobs 10, 20 and 30 and both earlier runs finished all of their steps with no traceback.

Among the probe runs, job 10 (16k head) was the first to get a final_eval val_loss below 3.28, at step 1315 (train_time 712892ms). Job 30 (8k head) reached it at step 1340 (train_time 731234ms). The seed-0 baseline, job 20, ended at 3.2788 after 1290 steps (train_time 769249ms).

## Results

| Run log | Seed | Job command | Final val_loss | Final train_time | First final_eval val_loss < 3.28 | step_avg @416 | step_avg @833 | step_avg @1250 | step_avg @last step | Peak memory | Traceback |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `run_10_adaptive16k_probe_seed0.log` | 0 | `TRAIN_SEED=0 NUM_EXTENSION_ITERATIONS=160 VAL_PROBE_EVERY=5 .venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt.py` | 3.2645 (step 1410/1410) | 784681ms | step 1315, val_loss 3.2796, train_time 712892ms | 302.20ms | 420.22ms | 530.87ms | 556.51ms (1410) | allocated 39587 MiB, reserved 50600 MiB | none |
| `run_20_baseline_seed0.log` | 0 | `TRAIN_SEED=0 .venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt_head.py` | 3.2788 (step 1290/1290) | 769249ms | not a probe run | 321.14ms | 453.23ms | 587.73ms | 596.32ms (1290) | allocated 39582 MiB, reserved 51240 MiB | none |
| `run_30_adaptive8k_probe_seed0.log` | 0 | `TRAIN_SEED=0 NUM_EXTENSION_ITERATIONS=160 VAL_PROBE_EVERY=5 ADAPTIVE_HEAD_TOKENS=8190 ADAPTIVE_CLUSTER_SIZES=16384 ADAPTIVE_TAIL_CAPACITY="3:0.45,0.28;2:0.37,0.22;1:0.13,0.07" .venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt_v2.py` | 3.2691 (step 1410/1410) | 783202ms | step 1340, val_loss 3.2799, train_time 731234ms | 309.25ms | 427.99ms | 531.44ms | 555.46ms (1410) | allocated 39587 MiB, reserved 46660 MiB | none |
| `run_40_adaptive8k1tail_fullfinal_probe_seed0.log` | 0 | `TRAIN_SEED=0 NUM_EXTENSION_ITERATIONS=160 VAL_PROBE_EVERY=5 ADAPTIVE_HEAD_TOKENS=8191 ADAPTIVE_CLUSTER_SIZES= ADAPTIVE_TAIL_CAPACITY="3:0.60;2:0.50;1:0.19" FULL_SOFTMAX_FROM_STAGE=2 .venv/bin/torchrun --standalone --nproc_per_node=1 train_gpt_v2.py` | none (crashed before step 0) | none | none (no val_loss lines) | none | none | none | none | not logged | `train_gpt_v2.py` line 2833 in warmup `model(...)`, from compiled graph `buf365 = empty_strided_cuda((262144, 50304), (50304, 1), torch.float32)`: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 49.12 GiB. GPU 0 has a total capacity of 79.19 GiB of which 39.53 GiB is free.` |
| `run_adaptive.log` | random (not logged) | not recorded (manual run of the first 16k adaptive model, 1290 steps) | 3.2846 (step 1290/1290) | 694389ms | not a probe run | 301.86ms | 420.22ms | 531.15ms | 538.29ms (1290) | allocated 39587 MiB, reserved 50600 MiB | none |
| `run_baseline.log` | random (not logged) | not recorded (manual baseline run, 1290 steps) | 3.2772 (step 1290/1290) | 769790ms | not a probe run | 321.32ms | 453.61ms | 588.13ms | 596.74ms (1290) | allocated 39582 MiB, reserved 51240 MiB | none |

## Vocab and adaptive softmax lines

The baseline runs (job 20 and `run_baseline.log`) print none of these lines. Job 40 crashed before printing any of them. Only job 30 printed an `adaptive softmax config` line.

| Run log | vocab stats | adaptive softmax coverage | adaptive softmax config | adaptive softmax tail rows |
|---|---|---|---|---|
| `run_10_adaptive16k_probe_seed0.log` | `vocab stats: counted 20,000,000 tokens of fineweb_train_000001.bin in 78ms, 895 token ids unseen`; `vocab stats: most frequent tokens ['.', ',', ' the', '\n', ' and', ' to', ' of', ' a']` | `adaptive softmax coverage: head 16382 tokens 93.45%, cluster 1 16384 tokens 4.97%, cluster 2 17491 tokens 1.58%` | not present | `adaptive softmax tail rows on rank 0: cluster 1 needed by 11.53% of rows, 0.000% of those over capacity, cluster 2 needed by 5.35% of rows, 0.000% of those over capacity` |
| `run_20_baseline_seed0.log` | not present | not present | not present | not present |
| `run_30_adaptive8k_probe_seed0.log` | `vocab stats: counted 20,000,000 tokens of fineweb_train_000001.bin in 125ms, 895 token ids unseen`; `vocab stats: most frequent tokens ['.', ',', ' the', '\n', ' and', ' to', ' of', ' a']` | `adaptive softmax coverage: head 8190 tokens 87.12%, cluster 1 16384 tokens 9.45%, cluster 2 25683 tokens 3.43%` | `adaptive softmax config: head_width 8192, clusters [(8192, 24576, 16384), (24576, 50304, 25683)], tail capacity {3: (0.45, 0.28), 2: (0.37, 0.22), 1: (0.13, 0.07)}, full softmax from stage -1` | `adaptive softmax tail rows on rank 0: cluster 1 needed by 19.35% of rows, 0.000% of those over capacity, cluster 2 needed by 9.76% of rows, 0.000% of those over capacity` |
| `run_40_adaptive8k1tail_fullfinal_probe_seed0.log` | not present | not present | not present | not present |
| `run_adaptive.log` | `vocab stats: counted 20,000,000 tokens of fineweb_train_000001.bin in 98ms, 895 token ids unseen`; `vocab stats: most frequent tokens ['.', ',', ' the', '\n', ' and', ' to', ' of', ' a']` | `adaptive softmax coverage: head 16382 tokens 93.45%, cluster 1 16384 tokens 4.97%, cluster 2 17491 tokens 1.58%` | not present | `adaptive softmax tail rows on rank 0: cluster 1 needed by 12.43% of rows, 0.000% of those over capacity, cluster 2 needed by 5.84% of rows, 0.000% of those over capacity` |
| `run_baseline.log` | not present | not present | not present | not present |

## final_eval lines from step 1250 on (probe runs)

The step 1250 validation line itself does not end in final_eval in either run. It reads `step:1250/1410 val_loss:3.2991 train_time:663699ms step_avg:530.96ms` for job 10 and `step:1250/1410 val_loss:3.3025 train_time:664410ms step_avg:531.53ms` for job 30. Job 40 has no final_eval lines.

### run_10_adaptive16k_probe_seed0.log

```
step:1255/1410 val_loss:3.2924 train_time:667544ms step_avg:531.91ms final_eval
step:1260/1410 val_loss:3.2908 train_time:671362ms step_avg:532.83ms final_eval
step:1265/1410 val_loss:3.2897 train_time:675131ms step_avg:533.70ms final_eval
step:1270/1410 val_loss:3.2882 train_time:678915ms step_avg:534.58ms final_eval
step:1275/1410 val_loss:3.2870 train_time:682686ms step_avg:535.44ms final_eval
step:1280/1410 val_loss:3.2858 train_time:686469ms step_avg:536.30ms final_eval
step:1285/1410 val_loss:3.2851 train_time:690237ms step_avg:537.15ms final_eval
step:1290/1410 val_loss:3.2840 train_time:694017ms step_avg:538.00ms final_eval
step:1295/1410 val_loss:3.2833 train_time:697785ms step_avg:538.83ms final_eval
step:1300/1410 val_loss:3.2821 train_time:701569ms step_avg:539.67ms final_eval
step:1305/1410 val_loss:3.2813 train_time:705337ms step_avg:540.49ms final_eval
step:1310/1410 val_loss:3.2804 train_time:709120ms step_avg:541.31ms final_eval
step:1315/1410 val_loss:3.2796 train_time:712892ms step_avg:542.12ms final_eval
step:1320/1410 val_loss:3.2787 train_time:716672ms step_avg:542.93ms final_eval
step:1325/1410 val_loss:3.2780 train_time:720443ms step_avg:543.73ms final_eval
step:1330/1410 val_loss:3.2770 train_time:724225ms step_avg:544.53ms final_eval
step:1335/1410 val_loss:3.2765 train_time:727994ms step_avg:545.31ms final_eval
step:1340/1410 val_loss:3.2756 train_time:731775ms step_avg:546.10ms final_eval
step:1345/1410 val_loss:3.2748 train_time:735552ms step_avg:546.88ms final_eval
step:1350/1410 val_loss:3.2741 train_time:739335ms step_avg:547.66ms final_eval
step:1355/1410 val_loss:3.2736 train_time:743110ms step_avg:548.42ms final_eval
step:1360/1410 val_loss:3.2725 train_time:746895ms step_avg:549.19ms final_eval
step:1365/1410 val_loss:3.2719 train_time:750669ms step_avg:549.94ms final_eval
step:1370/1410 val_loss:3.2710 train_time:754447ms step_avg:550.69ms final_eval
step:1375/1410 val_loss:3.2702 train_time:758222ms step_avg:551.43ms final_eval
step:1380/1410 val_loss:3.2691 train_time:762008ms step_avg:552.18ms final_eval
step:1385/1410 val_loss:3.2683 train_time:765781ms step_avg:552.91ms final_eval
step:1390/1410 val_loss:3.2674 train_time:769565ms step_avg:553.64ms final_eval
step:1395/1410 val_loss:3.2666 train_time:773337ms step_avg:554.36ms final_eval
step:1400/1410 val_loss:3.2658 train_time:777123ms step_avg:555.09ms final_eval
step:1405/1410 val_loss:3.2651 train_time:780899ms step_avg:555.80ms final_eval
step:1410/1410 val_loss:3.2645 train_time:784681ms step_avg:556.51ms final_eval
```

### run_30_adaptive8k_probe_seed0.log

```
step:1255/1410 val_loss:3.2967 train_time:668196ms step_avg:532.43ms final_eval
step:1260/1410 val_loss:3.2950 train_time:671897ms step_avg:533.25ms final_eval
step:1265/1410 val_loss:3.2939 train_time:675593ms step_avg:534.07ms final_eval
step:1270/1410 val_loss:3.2924 train_time:679309ms step_avg:534.89ms final_eval
step:1275/1410 val_loss:3.2914 train_time:683017ms step_avg:535.70ms final_eval
step:1280/1410 val_loss:3.2903 train_time:686726ms step_avg:536.51ms final_eval
step:1285/1410 val_loss:3.2894 train_time:690428ms step_avg:537.30ms final_eval
step:1290/1410 val_loss:3.2884 train_time:694140ms step_avg:538.09ms final_eval
step:1295/1410 val_loss:3.2876 train_time:697845ms step_avg:538.88ms final_eval
step:1300/1410 val_loss:3.2865 train_time:701557ms step_avg:539.66ms final_eval
step:1305/1410 val_loss:3.2859 train_time:705263ms step_avg:540.43ms final_eval
step:1310/1410 val_loss:3.2851 train_time:708978ms step_avg:541.20ms final_eval
step:1315/1410 val_loss:3.2839 train_time:712682ms step_avg:541.96ms final_eval
step:1320/1410 val_loss:3.2829 train_time:716394ms step_avg:542.72ms final_eval
step:1325/1410 val_loss:3.2823 train_time:720097ms step_avg:543.47ms final_eval
step:1330/1410 val_loss:3.2814 train_time:723815ms step_avg:544.22ms final_eval
step:1335/1410 val_loss:3.2809 train_time:727520ms step_avg:544.96ms final_eval
step:1340/1410 val_loss:3.2799 train_time:731234ms step_avg:545.70ms final_eval
step:1345/1410 val_loss:3.2792 train_time:734942ms step_avg:546.43ms final_eval
step:1350/1410 val_loss:3.2786 train_time:738659ms step_avg:547.15ms final_eval
step:1355/1410 val_loss:3.2778 train_time:742364ms step_avg:547.87ms final_eval
step:1360/1410 val_loss:3.2771 train_time:746081ms step_avg:548.59ms final_eval
step:1365/1410 val_loss:3.2764 train_time:749792ms step_avg:549.30ms final_eval
step:1370/1410 val_loss:3.2755 train_time:753507ms step_avg:550.01ms final_eval
step:1375/1410 val_loss:3.2747 train_time:757215ms step_avg:550.70ms final_eval
step:1380/1410 val_loss:3.2734 train_time:760933ms step_avg:551.40ms final_eval
step:1385/1410 val_loss:3.2729 train_time:764639ms step_avg:552.09ms final_eval
step:1390/1410 val_loss:3.2720 train_time:768353ms step_avg:552.77ms final_eval
step:1395/1410 val_loss:3.2711 train_time:772058ms step_avg:553.45ms final_eval
step:1400/1410 val_loss:3.2704 train_time:775778ms step_avg:554.13ms final_eval
step:1405/1410 val_loss:3.2696 train_time:779487ms step_avg:554.79ms final_eval
step:1410/1410 val_loss:3.2691 train_time:783202ms step_avg:555.46ms final_eval
```
