###### 1.nsl运行

Random seed set to: 5011
Current seed: 5011
shape of x_train  torch.Size([125973, 121])
shape of x_test is  torch.Size([22544, 121])
size of memory is  25194.599999999995
Traceback (most recent call last):
  File "/home/jackson/wkcode/SSF-Strategic-Selection-and-Forgetting-main/ssf.py", line 116, in <module>
    model = AE(input_dim).to(device)
  File "/home/jackson/anaconda3/envs/ssf/lib/python3.9/site-packages/torch/nn/modules/module.py", line 989, in to
    return self._apply(convert)
  File "/home/jackson/anaconda3/envs/ssf/lib/python3.9/site-packages/torch/nn/modules/module.py", line 641, in _apply
    module._apply(fn)
  File "/home/jackson/anaconda3/envs/ssf/lib/python3.9/site-packages/torch/nn/modules/module.py", line 641, in _apply
    module._apply(fn)
  File "/home/jackson/anaconda3/envs/ssf/lib/python3.9/site-packages/torch/nn/modules/module.py", line 664, in _apply
    param_applied = fn(param)
  File "/home/jackson/anaconda3/envs/ssf/lib/python3.9/site-packages/torch/nn/modules/module.py", line 987, in convert
    return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
RuntimeError: CUDA error: invalid device ordinal
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
(ssf) jackson@omnisky:~/wkcode/SSF-Strategic-Selection-and-Forgetting-main$ python ssf.py --cuda 0 --dataset nsl --epochs 200 --epoch_1 20 --sample_interval 5000 --num_labeled_sample 50 --opt_old_lr 100 --opt_new_lr 8 --new_sample_weight 3
Random seed set to: 5011
Current seed: 5011
shape of x_train  torch.Size([125973, 121])
shape of x_test is  torch.Size([22544, 121])
size of memory is  25194.599999999995
seed =  5011 , first round: epoch =  0
seed =  5011 , first round: epoch =  50
seed =  5011 , first round: epoch =  100
seed =  5011 , first round: epoch =  150
shape of x_test_left_epoch is  torch.Size([123323, 121])
--------------------------- performance_before_continual_training -----------------------------------
Confusion matrix
[[ 8218  1493]
 [ 1383 11450]]
Accuracy  0.8724272533711852
Precision  0.8846480723170826
Recall  0.8922309670381049
F1 score  0.888423339540658
seed =  5011 , i =  0
No drift detected in window 1 (p-value: 0.9746535577662471)
Selected representative old samples: torch.Size([25036, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  1
No drift detected in window 1 (p-value: 0.5824704180020326)
Selected representative old samples: torch.Size([25002, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  2
No drift detected in window 1 (p-value: 0.3581989308961231)
Selected representative old samples: torch.Size([25007, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  3
No drift detected in window 1 (p-value: 0.5237257069837198)
Selected representative old samples: torch.Size([24942, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  4
No drift detected in window 1 (p-value: 0.3897800357354626)
Selected representative old samples: torch.Size([24987, 121])
Selected representative new samples: torch.Size([5, 121])
Not enough representative new samples selected (5). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  5
No drift detected in window 1 (p-value: 0.41176566660913594)
Selected representative old samples: torch.Size([25111, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  6
No drift detected in window 1 (p-value: 0.20427030093735332)
Selected representative old samples: torch.Size([24928, 121])
Selected representative new samples: torch.Size([6, 121])
Not enough representative new samples selected (6). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  7
No drift detected in window 1 (p-value: 0.15120976750603699)
Selected representative old samples: torch.Size([24809, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  8
No drift detected in window 1 (p-value: 0.5188845158854625)
Selected representative old samples: torch.Size([25081, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  9
No drift detected in window 1 (p-value: 0.5817822800299504)
Selected representative old samples: torch.Size([25095, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  10
No drift detected in window 1 (p-value: 0.9986842642656035)
Selected representative old samples: torch.Size([25068, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  11
No drift detected in window 1 (p-value: 0.09862391674093429)
Selected representative old samples: torch.Size([24878, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  12
No drift detected in window 1 (p-value: 0.7940254909103336)
Selected representative old samples: torch.Size([24968, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  13
No drift detected in window 1 (p-value: 0.1402973379535698)
Selected representative old samples: torch.Size([24818, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  14
No drift detected in window 1 (p-value: 0.24576439643628012)
Selected representative old samples: torch.Size([24880, 121])
Selected representative new samples: torch.Size([5, 121])
Not enough representative new samples selected (5). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  15
No drift detected in window 1 (p-value: 0.6009465556824969)
Selected representative old samples: torch.Size([25060, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  16
No drift detected in window 1 (p-value: 0.08461166452626234)
Selected representative old samples: torch.Size([25091, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  17
No drift detected in window 1 (p-value: 0.42131529684764557)
Selected representative old samples: torch.Size([24887, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  18
No drift detected in window 1 (p-value: 0.7432851035653528)
Selected representative old samples: torch.Size([25010, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  19
No drift detected in window 1 (p-value: 0.765264953166112)
Selected representative old samples: torch.Size([25031, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5011 , i =  20
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 4.188521581133627e-16)
Drift detected, update the model...
Selected representative old samples: torch.Size([24964, 121])
Selected representative new samples: torch.Size([5, 121])
Not enough representative samples selected (5). Selecting additional random samples.
Buffer memory has extra space for 180 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5011 , i =  21
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 4.336174579377192e-17)
Drift detected, update the model...
Selected representative old samples: torch.Size([24576, 121])
Selected representative new samples: torch.Size([10, 121])
Not enough representative samples selected (10). Selecting additional random samples.
Buffer memory has extra space for 568 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5011 , i =  22
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 3.158970897375927e-12)
Drift detected, update the model...
Selected representative old samples: torch.Size([24327, 121])
Selected representative new samples: torch.Size([13, 121])
Not enough representative samples selected (13). Selecting additional random samples.
Buffer memory has extra space for 817 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5011 , i =  23
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 7.246286563675182e-08)
Drift detected, update the model...
Selected representative old samples: torch.Size([24254, 121])
Selected representative new samples: torch.Size([28, 121])
Not enough representative samples selected (28). Selecting additional random samples.
Buffer memory has extra space for 890 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5011 , i =  24
Selected representative old samples: torch.Size([24909, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[ 8279  1318]
 [  956 11754]]
Accuracy  0.8980589052763707
Precision  0.8991738066095472
Recall  0.9247836349331235
F1 score  0.9117989294856876
Random seed set to: 5012
Current seed: 5012
shape of x_train  torch.Size([125973, 121])
shape of x_test is  torch.Size([22544, 121])
size of memory is  25194.599999999995
seed =  5012 , first round: epoch =  0
seed =  5012 , first round: epoch =  50
seed =  5012 , first round: epoch =  100
seed =  5012 , first round: epoch =  150
shape of x_test_left_epoch is  torch.Size([123323, 121])
--------------------------- performance_before_continual_training -----------------------------------
Confusion matrix
[[8653 1058]
 [2842 9991]]
Accuracy  0.8270049680624556
Precision  0.9042447280296859
Recall  0.7785397023299306
F1 score  0.8366970940457248
seed =  5012 , i =  0
No drift detected in window 1 (p-value: 0.4090614690774902)
Selected representative old samples: torch.Size([24898, 121])
Selected representative new samples: torch.Size([6, 121])
Not enough representative new samples selected (6). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  1
No drift detected in window 1 (p-value: 0.732379272231114)
Selected representative old samples: torch.Size([25071, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  2
No drift detected in window 1 (p-value: 0.37057030653181344)
Selected representative old samples: torch.Size([25041, 121])
Selected representative new samples: torch.Size([4, 121])
Not enough representative new samples selected (4). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  3
No drift detected in window 1 (p-value: 0.482307638899069)
Selected representative old samples: torch.Size([24997, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  4
No drift detected in window 1 (p-value: 0.2599310908134932)
Selected representative old samples: torch.Size([24848, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  5
No drift detected in window 1 (p-value: 0.6818059924973325)
Selected representative old samples: torch.Size([25003, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  6
No drift detected in window 1 (p-value: 0.6687126258986251)
Selected representative old samples: torch.Size([25036, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  7
No drift detected in window 1 (p-value: 0.838885849118821)
Selected representative old samples: torch.Size([25099, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  8
No drift detected in window 1 (p-value: 0.9372984954186049)
Selected representative old samples: torch.Size([25075, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  9
No drift detected in window 1 (p-value: 0.17478616133484348)
Selected representative old samples: torch.Size([24808, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  10
No drift detected in window 1 (p-value: 0.6263738175666834)
Selected representative old samples: torch.Size([25010, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  11
No drift detected in window 1 (p-value: 0.05897914475772714)
Selected representative old samples: torch.Size([24847, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  12
No drift detected in window 1 (p-value: 0.5480783923136037)
Selected representative old samples: torch.Size([24998, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  13
No drift detected in window 1 (p-value: 0.8413580743122029)
Selected representative old samples: torch.Size([25088, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  14
No drift detected in window 1 (p-value: 0.9731351672321176)
Selected representative old samples: torch.Size([25044, 121])
Selected representative new samples: torch.Size([5, 121])
Not enough representative new samples selected (5). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  15
No drift detected in window 1 (p-value: 0.9237918201843811)
Selected representative old samples: torch.Size([25129, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  16
No drift detected in window 1 (p-value: 0.9329557262920256)
Selected representative old samples: torch.Size([25043, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  17
No drift detected in window 1 (p-value: 0.8770050590795738)
Selected representative old samples: torch.Size([25057, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  18
No drift detected in window 1 (p-value: 0.7902126531432913)
Selected representative old samples: torch.Size([25137, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  19
No drift detected in window 1 (p-value: 0.14625967292851472)
Selected representative old samples: torch.Size([24924, 121])
Selected representative new samples: torch.Size([5, 121])
Not enough representative new samples selected (5). Selecting additional random samples.
epoch =  0
seed =  5012 , i =  20
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 9.524167324757793e-18)
Drift detected, update the model...
Selected representative old samples: torch.Size([24928, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative samples selected (3). Selecting additional random samples.
Buffer memory has extra space for 216 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5012 , i =  21
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 6.756893534987153e-19)
Drift detected, update the model...
Selected representative old samples: torch.Size([24144, 121])
Selected representative new samples: torch.Size([14, 121])
Not enough representative samples selected (14). Selecting additional random samples.
Buffer memory has extra space for 1000 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5012 , i =  22
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 5.368141939442497e-08)
Drift detected, update the model...
Selected representative old samples: torch.Size([24552, 121])
Selected representative new samples: torch.Size([8, 121])
Not enough representative samples selected (8). Selecting additional random samples.
Buffer memory has extra space for 592 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5012 , i =  23
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 1.9859333003593453e-06)
Drift detected, update the model...
Selected representative old samples: torch.Size([24683, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative samples selected (3). Selecting additional random samples.
Buffer memory has extra space for 461 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5012 , i =  24
Selected representative old samples: torch.Size([24212, 121])
Selected representative new samples: torch.Size([21, 121])
Not enough representative new samples selected (21). Selecting additional random samples.
epoch =  0
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[ 8184  1432]
 [  880 11812]]
Accuracy  0.896360050206204
Precision  0.891875566294171
Recall  0.930664985817838
F1 score  0.9108574953732262
Random seed set to: 5013
Current seed: 5013
shape of x_train  torch.Size([125973, 121])
shape of x_test is  torch.Size([22544, 121])
size of memory is  25194.599999999995
seed =  5013 , first round: epoch =  0
seed =  5013 , first round: epoch =  50
seed =  5013 , first round: epoch =  100
seed =  5013 , first round: epoch =  150
shape of x_test_left_epoch is  torch.Size([123323, 121])
--------------------------- performance_before_continual_training -----------------------------------
Confusion matrix
[[ 8480  1231]
 [ 1795 11038]]
Accuracy  0.8657735982966643
Precision  0.8996658244355693
Recall  0.860126237045118
F1 score  0.8794518365070513
seed =  5013 , i =  0
No drift detected in window 1 (p-value: 0.41841150136136973)
Selected representative old samples: torch.Size([24984, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  1
No drift detected in window 1 (p-value: 0.14385732612023772)
Selected representative old samples: torch.Size([24932, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  2
No drift detected in window 1 (p-value: 0.46790635461746544)
Selected representative old samples: torch.Size([25134, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  3
No drift detected in window 1 (p-value: 0.7189991131009665)
Selected representative old samples: torch.Size([25037, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  4
No drift detected in window 1 (p-value: 0.7457717599962281)
Selected representative old samples: torch.Size([25069, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  5
No drift detected in window 1 (p-value: 0.47943176553198774)
Selected representative old samples: torch.Size([24987, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  6
No drift detected in window 1 (p-value: 0.4886456347425584)
Selected representative old samples: torch.Size([24990, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  7
No drift detected in window 1 (p-value: 0.8989004014263395)
Selected representative old samples: torch.Size([25138, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  8
No drift detected in window 1 (p-value: 0.24452323599245007)
Selected representative old samples: torch.Size([24963, 121])
Selected representative new samples: torch.Size([4, 121])
Not enough representative new samples selected (4). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  9
No drift detected in window 1 (p-value: 0.3734906408301394)
Selected representative old samples: torch.Size([25035, 121])
Selected representative new samples: torch.Size([4, 121])
Not enough representative new samples selected (4). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  10
No drift detected in window 1 (p-value: 0.9980006050545669)
Selected representative old samples: torch.Size([25118, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  11
No drift detected in window 1 (p-value: 0.12945280902450762)
Selected representative old samples: torch.Size([24981, 121])
Selected representative new samples: torch.Size([5, 121])
Not enough representative new samples selected (5). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  12
No drift detected in window 1 (p-value: 0.8170295951461636)
Selected representative old samples: torch.Size([25054, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  13
No drift detected in window 1 (p-value: 0.7528459699117619)
Selected representative old samples: torch.Size([25129, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  14
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 0.04898740768647003)
Drift detected, update the model...
Selected representative old samples: torch.Size([24858, 121])
Selected representative new samples: torch.Size([6, 121])
Not enough representative samples selected (6). Selecting additional random samples.
Buffer memory has extra space for 286 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5013 , i =  15
No drift detected in window 1 (p-value: 0.09739940622040477)
Selected representative old samples: torch.Size([25131, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  16
No drift detected in window 1 (p-value: 0.5946436289649244)
Selected representative old samples: torch.Size([25018, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  17
No drift detected in window 1 (p-value: 0.9985654744295506)
Selected representative old samples: torch.Size([25110, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  18
No drift detected in window 1 (p-value: 0.9284742384884017)
Selected representative old samples: torch.Size([25149, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough non-representative old samples to remove (45). Removing additional representative samples.
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  19
No drift detected in window 1 (p-value: 0.32071964178810874)
Selected representative old samples: torch.Size([24944, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5013 , i =  20
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 9.554747865013814e-12)
Drift detected, update the model...
Selected representative old samples: torch.Size([24067, 121])
Selected representative new samples: torch.Size([12, 121])
Not enough representative samples selected (12). Selecting additional random samples.
Buffer memory has extra space for 1077 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5013 , i =  21
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 7.026901628402922e-15)
Drift detected, update the model...
Selected representative old samples: torch.Size([24317, 121])
Selected representative new samples: torch.Size([7, 121])
Not enough representative samples selected (7). Selecting additional random samples.
Buffer memory has extra space for 827 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5013 , i =  22
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 0.0006509902673113696)
Drift detected, update the model...
Selected representative old samples: torch.Size([24868, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative samples selected (2). Selecting additional random samples.
Buffer memory has extra space for 276 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5013 , i =  23
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 1.9647499525771828e-05)
Drift detected, update the model...
Selected representative old samples: torch.Size([24446, 121])
Selected representative new samples: torch.Size([10, 121])
Not enough representative samples selected (10). Selecting additional random samples.
Buffer memory has extra space for 698 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5013 , i =  24
Selected representative old samples: torch.Size([24917, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[ 7902  1723]
 [  502 12188]]
Accuracy  0.9002912838897602
Precision  0.8761411832362879
Recall  0.960441292356186
F1 score  0.9163565279500772
Random seed set to: 5014
Current seed: 5014
shape of x_train  torch.Size([125973, 121])
shape of x_test is  torch.Size([22544, 121])
size of memory is  25194.599999999995
seed =  5014 , first round: epoch =  0
seed =  5014 , first round: epoch =  50
seed =  5014 , first round: epoch =  100
seed =  5014 , first round: epoch =  150
shape of x_test_left_epoch is  torch.Size([123323, 121])
--------------------------- performance_before_continual_training -----------------------------------
Confusion matrix
[[ 8559  1152]
 [ 2417 10416]]
Accuracy  0.8416873669268985
Precision  0.9004149377593361
Recall  0.811657445647939
F1 score  0.8537355026433343
seed =  5014 , i =  0
No drift detected in window 1 (p-value: 0.9959364910814992)
Selected representative old samples: torch.Size([25116, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  1
No drift detected in window 1 (p-value: 0.37526109044483824)
Selected representative old samples: torch.Size([25046, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  2
No drift detected in window 1 (p-value: 0.32022335486502573)
Selected representative old samples: torch.Size([24931, 121])
Selected representative new samples: torch.Size([4, 121])
Not enough representative new samples selected (4). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  3
No drift detected in window 1 (p-value: 0.9987201992074245)
Selected representative old samples: torch.Size([25074, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  4
No drift detected in window 1 (p-value: 0.7039628024247365)
Selected representative old samples: torch.Size([25027, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  5
No drift detected in window 1 (p-value: 0.8782103062612168)
Selected representative old samples: torch.Size([25124, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  6
No drift detected in window 1 (p-value: 0.6106138385807791)
Selected representative old samples: torch.Size([25085, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  7
No drift detected in window 1 (p-value: 0.9810418658480098)
Selected representative old samples: torch.Size([25097, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  8
No drift detected in window 1 (p-value: 0.781823442025503)
Selected representative old samples: torch.Size([25076, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  9
No drift detected in window 1 (p-value: 0.2610996232912569)
Selected representative old samples: torch.Size([25142, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  10
No drift detected in window 1 (p-value: 0.7016805551509759)
Selected representative old samples: torch.Size([25078, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  11
No drift detected in window 1 (p-value: 0.2526828628036085)
Selected representative old samples: torch.Size([25006, 121])
Selected representative new samples: torch.Size([5, 121])
Not enough representative new samples selected (5). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  12
No drift detected in window 1 (p-value: 0.608785523121424)
Selected representative old samples: torch.Size([25011, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  13
No drift detected in window 1 (p-value: 0.88709339508872)
Selected representative old samples: torch.Size([25096, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  14
No drift detected in window 1 (p-value: 0.966130647930213)
Selected representative old samples: torch.Size([25052, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  15
No drift detected in window 1 (p-value: 0.5856207189398703)
Selected representative old samples: torch.Size([25052, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  16
No drift detected in window 1 (p-value: 0.5620022711771343)
Selected representative old samples: torch.Size([25060, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  17
No drift detected in window 1 (p-value: 0.22716761954788078)
Selected representative old samples: torch.Size([24981, 121])
Selected representative new samples: torch.Size([6, 121])
Not enough representative new samples selected (6). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  18
No drift detected in window 1 (p-value: 0.7574397293980707)
Selected representative old samples: torch.Size([25085, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  19
No drift detected in window 1 (p-value: 0.9548751141590239)
Selected representative old samples: torch.Size([25121, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5014 , i =  20
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 4.2876325868369047e-11)
Drift detected, update the model...
Selected representative old samples: torch.Size([24655, 121])
Selected representative new samples: torch.Size([10, 121])
Not enough representative samples selected (10). Selecting additional random samples.
Buffer memory has extra space for 489 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5014 , i =  21
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 7.977700561125719e-19)
Drift detected, update the model...
Selected representative old samples: torch.Size([24595, 121])
Selected representative new samples: torch.Size([6, 121])
Not enough representative samples selected (6). Selecting additional random samples.
Buffer memory has extra space for 549 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5014 , i =  22
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 3.860062090277125e-08)
Drift detected, update the model...
Selected representative old samples: torch.Size([24349, 121])
Selected representative new samples: torch.Size([10, 121])
Not enough representative samples selected (10). Selecting additional random samples.
Buffer memory has extra space for 795 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5014 , i =  23
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 6.694656789237663e-08)
Drift detected, update the model...
Selected representative old samples: torch.Size([24381, 121])
Selected representative new samples: torch.Size([10, 121])
Not enough representative samples selected (10). Selecting additional random samples.
Buffer memory has extra space for 763 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5014 , i =  24
Selected representative old samples: torch.Size([24840, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[ 8094  1520]
 [  630 12069]]
Accuracy  0.9036436158293372
Precision  0.888144823018618
Recall  0.9503897944720057
F1 score  0.9182136335970784
Random seed set to: 5015
Current seed: 5015
shape of x_train  torch.Size([125973, 121])
shape of x_test is  torch.Size([22544, 121])
size of memory is  25194.599999999995
seed =  5015 , first round: epoch =  0
seed =  5015 , first round: epoch =  50
seed =  5015 , first round: epoch =  100
seed =  5015 , first round: epoch =  150
shape of x_test_left_epoch is  torch.Size([123323, 121])
--------------------------- performance_before_continual_training -----------------------------------
Confusion matrix
[[ 8145  1566]
 [ 1987 10846]]
Accuracy  0.8423970901348474
Precision  0.8738317757009346
Recall  0.8451648094755708
F1 score  0.8592592592592593
seed =  5015 , i =  0
No drift detected in window 1 (p-value: 0.6386459248424603)
Selected representative old samples: torch.Size([25032, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  1
No drift detected in window 1 (p-value: 0.08028698899058517)
Selected representative old samples: torch.Size([24923, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  2
No drift detected in window 1 (p-value: 0.9999211597367985)
Selected representative old samples: torch.Size([25151, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough non-representative old samples to remove (43). Removing additional representative samples.
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  3
No drift detected in window 1 (p-value: 0.3155188985524563)
Selected representative old samples: torch.Size([24970, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  4
No drift detected in window 1 (p-value: 0.8917253623400472)
Selected representative old samples: torch.Size([25132, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  5
No drift detected in window 1 (p-value: 0.06889773598300286)
Selected representative old samples: torch.Size([24795, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  6
No drift detected in window 1 (p-value: 0.47050512418937174)
Selected representative old samples: torch.Size([24936, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  7
No drift detected in window 1 (p-value: 0.08727782340834445)
Selected representative old samples: torch.Size([24927, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  8
No drift detected in window 1 (p-value: 0.2622978293841336)
Selected representative old samples: torch.Size([24958, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  9
No drift detected in window 1 (p-value: 0.702237596251975)
Selected representative old samples: torch.Size([25058, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  10
No drift detected in window 1 (p-value: 0.5299110125693186)
Selected representative old samples: torch.Size([25034, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  11
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 0.002391611826349723)
Drift detected, update the model...
Selected representative old samples: torch.Size([24715, 121])
Selected representative new samples: torch.Size([4, 121])
Not enough representative samples selected (4). Selecting additional random samples.
Buffer memory has extra space for 429 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5015 , i =  12
No drift detected in window 1 (p-value: 0.99602073024288)
Selected representative old samples: torch.Size([25117, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  13
No drift detected in window 1 (p-value: 0.0816935057452598)
Selected representative old samples: torch.Size([24918, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  14
No drift detected in window 1 (p-value: 0.8170509403983306)
Selected representative old samples: torch.Size([25047, 121])
Selected representative new samples: torch.Size([1, 121])
Not enough representative new samples selected (1). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  15
No drift detected in window 1 (p-value: 0.6908093186869213)
Selected representative old samples: torch.Size([25098, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  16
No drift detected in window 1 (p-value: 0.6975222611112836)
Selected representative old samples: torch.Size([25088, 121])
Selected representative new samples: torch.Size([2, 121])
Not enough representative new samples selected (2). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  17
No drift detected in window 1 (p-value: 0.6464040911164548)
Selected representative old samples: torch.Size([25144, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  18
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 0.03183728972557387)
Drift detected, update the model...
Selected representative old samples: torch.Size([24859, 121])
Selected representative new samples: torch.Size([8, 121])
Not enough representative samples selected (8). Selecting additional random samples.
Buffer memory has extra space for 285 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5015 , i =  19
No drift detected in window 1 (p-value: 0.8898814478691611)
Selected representative old samples: torch.Size([25096, 121])
Selected representative new samples: torch.Size([3, 121])
Not enough representative new samples selected (3). Selecting additional random samples.
epoch =  0
seed =  5015 , i =  20
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 1.3119085325638618e-09)
Drift detected, update the model...
Selected representative old samples: torch.Size([24211, 121])
Selected representative new samples: torch.Size([34, 121])
Not enough representative samples selected (34). Selecting additional random samples.
Buffer memory has extra space for 933 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5015 , i =  21
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 2.2255449955571212e-30)
Drift detected, update the model...
Selected representative old samples: torch.Size([23868, 121])
Selected representative new samples: torch.Size([11, 121])
Not enough representative samples selected (11). Selecting additional random samples.
Buffer memory has extra space for 1276 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5015 , i =  22
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 0.008561594339530146)
Drift detected, update the model...
Selected representative old samples: torch.Size([25050, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative samples selected (0). Selecting additional random samples.
Buffer memory has extra space for 94 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5015 , i =  23
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 2.7274503705392577e-92)
Drift detected, update the model...
Selected representative old samples: torch.Size([24290, 121])
Selected representative new samples: torch.Size([22, 121])
Not enough representative samples selected (22). Selecting additional random samples.
Buffer memory has extra space for 854 samples. Adding new samples with pseudo labels.
epoch =  0
seed =  5015 , i =  24
Selected representative old samples: torch.Size([25048, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
epoch =  0
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[ 7934  1687]
 [  444 12265]]
Accuracy  0.9045678459471563
Precision  0.8790854357798165
Recall  0.9650641277834605
F1 score  0.9200705149844342




###### 2.usw
epoch =  50
epoch =  100
epoch =  150
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[30999  5631]
 [ 2865 42015]]
Accuracy  0.8957673905042326
Precision  0.881815892205012
Recall  0.9361631016042781
F1 score  0.9081771610142014
Random seed set to: 5013
Current seed: 5013
shape of x_train  torch.Size([175341, 196])
shape of x_test is  torch.Size([82332, 196])
size of memory is  35068.19999999999
seed =  5013 , first round: epoch =  0
seed =  5013 , first round: epoch =  50
seed =  5013 , first round: epoch =  100
seed =  5013 , first round: epoch =  150
shape of x_test_left_epoch is  torch.Size([222605, 196])
--------------------------- performance_before_contunual_training -----------------------------------
Confusion matrix
[[22465 14535]
 [  466 44866]]
Accuracy  0.8177986688043531
Precision  0.7553071497112843
Recall  0.9897202858907614
F1 score  0.8567691176611
seed =  5013 , i =  0
No drift detected in window 1 (p-value: 0.8470831849132372)
Selected representative old samples: torch.Size([35013, 196])
Selected representative new samples: torch.Size([46, 196])
Not enough non-representative old samples to remove (55). Removing additional representative samples.
Not enough representative new samples selected (46). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5013 , i =  1
No drift detected in window 1 (p-value: 0.49146417242979823)
Selected representative old samples: torch.Size([35026, 196])
Selected representative new samples: torch.Size([49, 196])
Not enough non-representative old samples to remove (42). Removing additional representative samples.
Not enough representative new samples selected (49). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5013 , i =  2
No drift detected in window 1 (p-value: 0.27264662985075594)
Selected representative old samples: torch.Size([35018, 196])
Selected representative new samples: torch.Size([50, 196])
Not enough non-representative old samples to remove (50). Removing additional representative samples.
Not enough representative new samples selected (50). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5013 , i =  3
No drift detected in window 1 (p-value: 0.7263862397568396)
Selected representative old samples: torch.Size([35004, 196])
Selected representative new samples: torch.Size([36, 196])
Not enough non-representative old samples to remove (64). Removing additional representative samples.
Not enough representative new samples selected (36). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5013 , i =  4
No drift detected in window 1 (p-value: 0.5661522295032071)
Selected representative old samples: torch.Size([35040, 196])
Selected representative new samples: torch.Size([23, 196])
Not enough non-representative old samples to remove (28). Removing additional representative samples.
Not enough representative new samples selected (23). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5013 , i =  5
No drift detected in window 1 (p-value: 0.55558910905199)
Selected representative old samples: torch.Size([35027, 196])
Selected representative new samples: torch.Size([22, 196])
Not enough non-representative old samples to remove (41). Removing additional representative samples.
Not enough representative new samples selected (22). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5013 , i =  6
No drift detected in window 1 (p-value: 0.9541812227659251)
Selected representative old samples: torch.Size([35041, 196])
Selected representative new samples: torch.Size([16, 196])
Not enough non-representative old samples to remove (27). Removing additional representative samples.
Not enough representative new samples selected (16). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5013 , i =  7
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 2.906146723425547e-134)
Drift detected, update the model...
Selected representative old samples: torch.Size([34701, 196])
Selected representative new samples: torch.Size([355, 196])
Buffer memory has extra space for 167 samples. Adding new samples with pseudo labels.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5013 , i =  8
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 7.664568378444536e-152)
Drift detected, update the model...
Selected representative old samples: torch.Size([34601, 196])
Selected representative new samples: torch.Size([436, 196])
Buffer memory has extra space for 267 samples. Adding new samples with pseudo labels.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5013 , i =  9
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 9.442735045112169e-111)
Drift detected, update the model...
Selected representative old samples: torch.Size([34721, 196])
Selected representative new samples: torch.Size([315, 196])
Buffer memory has extra space for 147 samples. Adding new samples with pseudo labels.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5013 , i =  10
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 5.4000994535789e-94)
Drift detected, update the model...
Selected representative old samples: torch.Size([34770, 196])
Selected representative new samples: torch.Size([254, 196])
Buffer memory has extra space for 98 samples. Adding new samples with pseudo labels.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5013 , i =  11
Selected representative old samples: torch.Size([34898, 196])
Selected representative new samples: torch.Size([36, 196])
Not enough non-representative old samples to remove (170). Removing additional representative samples.
Not enough representative new samples selected (36). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[30272  6330]
 [ 1999 42892]]
Accuracy  0.8977949026296738
Precision  0.8713989679411646
Recall  0.955469916018801
F1 score  0.9115000053127623
Random seed set to: 5014
Current seed: 5014
shape of x_train  torch.Size([175341, 196])
shape of x_test is  torch.Size([82332, 196])
size of memory is  35068.19999999999
seed =  5014 , first round: epoch =  0
seed =  5014 , first round: epoch =  50
seed =  5014 , first round: epoch =  100
seed =  5014 , first round: epoch =  150
shape of x_test_left_epoch is  torch.Size([222605, 196])
--------------------------- performance_before_contunual_training -----------------------------------
Confusion matrix
[[22212 14788]
 [  298 45034]]
Accuracy  0.81676626342127
Precision  0.7527999732539868
Recall  0.9934262772434483
F1 score  0.8565342259923541
seed =  5014 , i =  0
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 0.021418486687893454)
Drift detected, update the model...
Selected representative old samples: torch.Size([35026, 196])
Selected representative new samples: torch.Size([38, 196])
Not enough non-representative old samples to remove (42). Removing additional representative samples.
Not enough representative samples selected (38). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5014 , i =  1
No drift detected in window 1 (p-value: 0.12636638838316627)
Selected representative old samples: torch.Size([35001, 196])
Selected representative new samples: torch.Size([45, 196])
Not enough non-representative old samples to remove (67). Removing additional representative samples.
Not enough representative new samples selected (45). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5014 , i =  2
No drift detected in window 1 (p-value: 0.7328799175406748)
Selected representative old samples: torch.Size([35007, 196])
Selected representative new samples: torch.Size([46, 196])
Not enough non-representative old samples to remove (61). Removing additional representative samples.
Not enough representative new samples selected (46). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5014 , i =  3
No drift detected in window 1 (p-value: 0.4221639665979108)
Selected representative old samples: torch.Size([35037, 196])
Selected representative new samples: torch.Size([31, 196])
Not enough non-representative old samples to remove (31). Removing additional representative samples.
Not enough representative new samples selected (31). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5014 , i =  4
No drift detected in window 1 (p-value: 0.7940353065309815)
Selected representative old samples: torch.Size([35031, 196])
Selected representative new samples: torch.Size([28, 196])
Not enough non-representative old samples to remove (37). Removing additional representative samples.
Not enough representative new samples selected (28). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5014 , i =  5
No drift detected in window 1 (p-value: 0.9948351408051713)
Selected representative old samples: torch.Size([35036, 196])
Selected representative new samples: torch.Size([24, 196])
Not enough non-representative old samples to remove (32). Removing additional representative samples.
Not enough representative new samples selected (24). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5014 , i =  6
No drift detected in window 1 (p-value: 0.5154717309894212)
Selected representative old samples: torch.Size([35030, 196])
Selected representative new samples: torch.Size([27, 196])
Not enough non-representative old samples to remove (38). Removing additional representative samples.
Not enough representative new samples selected (27). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5014 , i =  7
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 2.7152807996916486e-134)
Drift detected, update the model...
Selected representative old samples: torch.Size([34695, 196])
Selected representative new samples: torch.Size([324, 196])
Buffer memory has extra space for 173 samples. Adding new samples with pseudo labels.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5014 , i =  8
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 2.3574624300155032e-138)
Drift detected, update the model...
Selected representative old samples: torch.Size([34638, 196])
Selected representative new samples: torch.Size([386, 196])
Buffer memory has extra space for 230 samples. Adding new samples with pseudo labels.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5014 , i =  9
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 3.989281489732316e-89)
Drift detected, update the model...
Selected representative old samples: torch.Size([34867, 196])
Selected representative new samples: torch.Size([158, 196])
Not enough representative samples selected (158). Selecting additional random samples.
Buffer memory has extra space for 1 samples. Adding new samples with pseudo labels.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5014 , i =  10
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 1.371763583023638e-106)
Drift detected, update the model...
Selected representative old samples: torch.Size([34823, 196])
Selected representative new samples: torch.Size([192, 196])
Not enough representative samples selected (192). Selecting additional random samples.
Buffer memory has extra space for 45 samples. Adding new samples with pseudo labels.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5014 , i =  11
Selected representative old samples: torch.Size([34862, 196])
Selected representative new samples: torch.Size([38, 196])
Not enough representative new samples selected (38). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[29844  6790]
 [ 1798 43070]]
Accuracy  0.8946283526784619
Precision  0.8638186923385479
Recall  0.9599268966746902
F1 score  0.9093404273287728
Random seed set to: 5015
Current seed: 5015
shape of x_train  torch.Size([175341, 196])
shape of x_test is  torch.Size([82332, 196])
size of memory is  35068.19999999999
seed =  5015 , first round: epoch =  0
seed =  5015 , first round: epoch =  50
seed =  5015 , first round: epoch =  100
seed =  5015 , first round: epoch =  150
shape of x_test_left_epoch is  torch.Size([222605, 196])
--------------------------- performance_before_contunual_training -----------------------------------
Confusion matrix
[[22490 14510]
 [  448 44884]]
Accuracy  0.8183209444687364
Precision  0.7556992288783378
Recall  0.990117356392835
F1 score  0.8571701392204418
seed =  5015 , i =  0
No drift detected in window 1 (p-value: 0.8775800742884111)
Selected representative old samples: torch.Size([35010, 196])
Selected representative new samples: torch.Size([29, 196])
Not enough non-representative old samples to remove (58). Removing additional representative samples.
Not enough representative new samples selected (29). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5015 , i =  1
No drift detected in window 1 (p-value: 0.6043859444204474)
Selected representative old samples: torch.Size([35015, 196])
Selected representative new samples: torch.Size([29, 196])
Not enough non-representative old samples to remove (53). Removing additional representative samples.
Not enough representative new samples selected (29). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5015 , i =  2
No drift detected in window 1 (p-value: 0.8692687270556866)
Selected representative old samples: torch.Size([35035, 196])
Selected representative new samples: torch.Size([26, 196])
Not enough non-representative old samples to remove (33). Removing additional representative samples.
Not enough representative new samples selected (26). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5015 , i =  3
No drift detected in window 1 (p-value: 0.4096601808045054)
Selected representative old samples: torch.Size([35026, 196])
Selected representative new samples: torch.Size([33, 196])
Not enough non-representative old samples to remove (42). Removing additional representative samples.
Not enough representative new samples selected (33). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5015 , i =  4
No drift detected in window 1 (p-value: 0.3781917248569362)
Selected representative old samples: torch.Size([35035, 196])
Selected representative new samples: torch.Size([28, 196])
Not enough non-representative old samples to remove (33). Removing additional representative samples.
Not enough representative new samples selected (28). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5015 , i =  5
No drift detected in window 1 (p-value: 0.7829636406956376)
Selected representative old samples: torch.Size([35009, 196])
Selected representative new samples: torch.Size([48, 196])
Not enough non-representative old samples to remove (59). Removing additional representative samples.
Not enough representative new samples selected (48). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5015 , i =  6
No drift detected in window 1 (p-value: 0.2044674479401093)
Selected representative old samples: torch.Size([34911, 196])
Selected representative new samples: torch.Size([134, 196])
Not enough non-representative old samples to remove (157). Removing additional representative samples.
Not enough representative new samples selected (134). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5015 , i =  7
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 1.82494704897607e-79)
Drift detected, update the model...
Selected representative old samples: torch.Size([34745, 196])
Selected representative new samples: torch.Size([298, 196])
Buffer memory has extra space for 123 samples. Adding new samples with pseudo labels.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5015 , i =  8
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 1.6787927688483112e-91)
Drift detected, update the model...
Selected representative old samples: torch.Size([34654, 196])
Selected representative new samples: torch.Size([415, 196])
Buffer memory has extra space for 214 samples. Adding new samples with pseudo labels.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5015 , i =  9
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 3.02957393912603e-44)
Drift detected, update the model...
Selected representative old samples: torch.Size([34829, 196])
Selected representative new samples: torch.Size([202, 196])
Buffer memory has extra space for 39 samples. Adding new samples with pseudo labels.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5015 , i =  10
!!!!!!!!!!!!!!!!!!!!! Drift detected in window 1 (p-value: 9.434150039157308e-50)
Drift detected, update the model...
Selected representative old samples: torch.Size([34828, 196])
Selected representative new samples: torch.Size([192, 196])
Not enough representative samples selected (192). Selecting additional random samples.
Buffer memory has extra space for 40 samples. Adding new samples with pseudo labels.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
seed =  5015 , i =  11
Selected representative old samples: torch.Size([34833, 196])
Selected representative new samples: torch.Size([46, 196])
Not enough representative new samples selected (46). Selecting additional random samples.
epoch =  0
epoch =  50
epoch =  100
epoch =  150
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[28859  7760]
 [ 1267 43629]]
Accuracy  0.889259645464025
Precision  0.8489949210920625
Recall  0.9717792230933714
F1 score  0.9062470789842655
