# Phase-9.4.4 Training Curve Audit

Phase-9.4.3 未保存中间 checkpoint。epoch 曲线中的 `train_loss` 来自正式训练日志；
validation / accuracy / gradient / entropy 只能在 **last-epoch 冻结权重** 上计算。
未重训、未改学习率、未覆盖 9.4.3 权重。

## deepsets
- seed 0: epoch1=8.9298 mid=0.8460 last=0.6012 (n=80)
- seed 1: epoch1=8.9388 mid=1.0289 last=0.5916 (n=80)
- seed 2: epoch1=8.9728 mid=0.7904 last=0.5904 (n=80)
- seed 3: epoch1=9.1094 mid=0.8507 last=0.6297 (n=80)
- seed 4: epoch1=9.1734 mid=0.8302 last=0.6850 (n=80)

## hgnn
- seed 0: epoch1=8.7718 mid=0.8386 last=0.7179 (n=80)
- seed 1: epoch1=8.8574 mid=1.0168 last=0.7624 (n=80)
- seed 2: epoch1=8.8739 mid=0.8036 last=0.6342 (n=80)
- seed 3: epoch1=8.8881 mid=0.9042 last=0.6770 (n=80)
- seed 4: epoch1=9.1265 mid=0.8779 last=0.6570 (n=80)

## last-epoch snapshot (train subsample / val subsample)
- deepsets seed=0: train_nll=0.5091 val_nll=1.9411 tf_acc=0.8681 fr_acc=0.8229 exact=0.6667 grad_l2=7.6631 param_l2=42.54
- deepsets seed=1: train_nll=0.5506 val_nll=3.0985 tf_acc=0.8646 fr_acc=0.8229 exact=0.6042 grad_l2=7.6039 param_l2=42.10
- deepsets seed=2: train_nll=0.4220 val_nll=2.5877 tf_acc=0.8924 fr_acc=0.8542 exact=0.6667 grad_l2=32.2858 param_l2=42.29
- deepsets seed=3: train_nll=0.4166 val_nll=2.2330 tf_acc=0.8924 fr_acc=0.8229 exact=0.6042 grad_l2=14.6648 param_l2=41.95
- deepsets seed=4: train_nll=0.4056 val_nll=2.1341 tf_acc=0.8889 fr_acc=0.8160 exact=0.6042 grad_l2=9.2935 param_l2=41.29
- hgnn seed=0: train_nll=0.7260 val_nll=2.3956 tf_acc=0.8785 fr_acc=0.8368 exact=0.6250 grad_l2=17.9597 param_l2=56.92
- hgnn seed=1: train_nll=0.7024 val_nll=2.7618 tf_acc=0.8507 fr_acc=0.8090 exact=0.6042 grad_l2=11.9497 param_l2=56.38
- hgnn seed=2: train_nll=0.4360 val_nll=1.9587 tf_acc=0.8819 fr_acc=0.8160 exact=0.6458 grad_l2=22.7211 param_l2=58.03
- hgnn seed=3: train_nll=0.4182 val_nll=2.4252 tf_acc=0.8785 fr_acc=0.8264 exact=0.6042 grad_l2=27.9266 param_l2=57.80
- hgnn seed=4: train_nll=0.3894 val_nll=1.4425 tf_acc=0.9097 fr_acc=0.8681 exact=0.6667 grad_l2=23.8322 param_l2=55.88

## HGNN positive (0/1/2) vs negative (3/4) last-epoch
- train NLL pos=0.6215 neg=0.4038
- val NLL pos=2.3720 neg=1.9338
- FR acc pos=0.8206 neg=0.8472
- grad L2 pos=17.5435 neg=25.8794

负向 payload seed 的监督指标更好，因此不能称为 optimization failure。
