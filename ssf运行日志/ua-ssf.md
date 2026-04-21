# 1.nsl
(ssf) jackson@omnisky:~/wkcodenew/SSF-Strategic-Selection-and-Forgetting-main$ python ssf.py --dataset nsl --epochs 200 --epoch_1 20 --sample_interval 5000 --num_labeled_sample 50 --opt_old_lr 100 --opt_new_lr 8 --new_sample_weight 3 --cuda 0
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
[[ 8221  1490]
 [ 1232 11601]]
Accuracy  0.8792583392476934
Precision  0.8861813459628752
Recall  0.9039975064287384
F1 score  0.8950007714858818
seed =  5011 , i =  0
No drift in window 1 (p-value: 0.972955, severity: 0.0000)
Selected representative old samples: torch.Size([118, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5011 , i =  1
No drift in window 1 (p-value: 0.537443, severity: 0.0000)
Selected representative old samples: torch.Size([88, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5011 , i =  2
No drift in window 1 (p-value: 0.613190, severity: 0.0000)
Selected representative old samples: torch.Size([401, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5011 , i =  3
No drift in window 1 (p-value: 0.576828, severity: 0.0000)
Selected representative old samples: torch.Size([70, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5011 , i =  4
No drift in window 1 (p-value: 0.564541, severity: 0.0000)
Selected representative old samples: torch.Size([229, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5011 , i =  5
!!! Drift in window 1 (p-value: 0.017286, severity: 0.1762)
Drift detected, update the model...
Selected representative old samples: torch.Size([168, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 24976 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.4119, epochs=13, severity=0.1762
epoch =  0
seed =  5011 , i =  6
No drift in window 1 (p-value: 0.068323, severity: 0.0000)
Selected representative old samples: torch.Size([5182, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5011 , i =  7
!!! Drift in window 1 (p-value: 0.022539, severity: 0.1647)
Drift detected, update the model...
Selected representative old samples: torch.Size([4711, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 20433 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.4176, epochs=13, severity=0.1647
epoch =  0
seed =  5011 , i =  8
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([6220, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 18924 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5011 , i =  9
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([6374, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 18770 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5011 , i =  10
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([5384, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 19760 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5011 , i =  11
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([0, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative samples selected (0). Selecting additional random samples.
Buffer memory has extra space for 25144 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5011 , i =  12
No drift in window 1 (p-value: 0.201739, severity: 0.0000)
Selected representative old samples: torch.Size([2178, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5011 , i =  13
!!! Drift in window 1 (p-value: 0.012273, severity: 0.1911)
Drift detected, update the model...
Selected representative old samples: torch.Size([2165, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 22979 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.4044, epochs=13, severity=0.1911
epoch =  0
seed =  5011 , i =  14
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([5490, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 19654 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5011 , i =  15
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([7304, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 17840 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5011 , i =  16
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([6847, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 18297 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5011 , i =  17
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([8227, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 16917 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5011 , i =  18
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([4736, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 20408 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5011 , i =  19
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([5200, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 19944 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5011 , i =  20
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([4626, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 20518 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5011 , i =  21
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([7508, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 17636 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5011 , i =  22
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([0, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative samples selected (0). Selecting additional random samples.
Buffer memory has extra space for 25144 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5011 , i =  23
No drift in window 1 (p-value: 0.564065, severity: 0.0000)
Selected representative old samples: torch.Size([30, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5011 , i =  24
Selected representative old samples: torch.Size([67, 121])
Selected representative new samples: torch.Size([3323, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[7884 1784]
 [4348 8301]]
Accuracy  0.7252318860061836
Precision  0.8231036192364899
Recall  0.6562574116530951
F1 score  0.7302718395354976
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
[[8744  967]
 [2964 9869]]
Accuracy  0.8256298793470547
Precision  0.9107604282022886
Recall  0.7690329618951142
F1 score  0.8339177827538129
seed =  5012 , i =  0
No drift in window 1 (p-value: 0.301692, severity: 0.0000)
Selected representative old samples: torch.Size([371, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  1
No drift in window 1 (p-value: 0.711051, severity: 0.0000)
Selected representative old samples: torch.Size([117, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  2
No drift in window 1 (p-value: 0.701790, severity: 0.0000)
Selected representative old samples: torch.Size([96, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  3
No drift in window 1 (p-value: 0.628848, severity: 0.0000)
Selected representative old samples: torch.Size([315, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  4
No drift in window 1 (p-value: 0.551186, severity: 0.0000)
Selected representative old samples: torch.Size([82, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  5
No drift in window 1 (p-value: 0.364118, severity: 0.0000)
Selected representative old samples: torch.Size([281, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  6
No drift in window 1 (p-value: 0.996561, severity: 0.0000)
Selected representative old samples: torch.Size([99, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  7
No drift in window 1 (p-value: 0.401606, severity: 0.0000)
Selected representative old samples: torch.Size([112, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  8
No drift in window 1 (p-value: 0.733091, severity: 0.0000)
Selected representative old samples: torch.Size([265, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  9
No drift in window 1 (p-value: 0.679008, severity: 0.0000)
Selected representative old samples: torch.Size([288, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  10
No drift in window 1 (p-value: 0.131909, severity: 0.0000)
Selected representative old samples: torch.Size([220, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  11
No drift in window 1 (p-value: 0.734810, severity: 0.0000)
Selected representative old samples: torch.Size([125, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  12
No drift in window 1 (p-value: 0.065679, severity: 0.0000)
Selected representative old samples: torch.Size([151, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  13
No drift in window 1 (p-value: 0.590080, severity: 0.0000)
Selected representative old samples: torch.Size([81, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  14
No drift in window 1 (p-value: 0.105645, severity: 0.0000)
Selected representative old samples: torch.Size([88, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  15
No drift in window 1 (p-value: 0.330765, severity: 0.0000)
Selected representative old samples: torch.Size([200, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  16
No drift in window 1 (p-value: 0.402886, severity: 0.0000)
Selected representative old samples: torch.Size([202, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  17
No drift in window 1 (p-value: 0.148869, severity: 0.0000)
Selected representative old samples: torch.Size([93, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5012 , i =  18
!!! Drift in window 1 (p-value: 0.010850, severity: 0.1965)
Drift detected, update the model...
Selected representative old samples: torch.Size([11648, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 13496 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.4018, epochs=13, severity=0.1965
epoch =  0
seed =  5012 , i =  19
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([7716, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 17428 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5012 , i =  20
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([7457, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 17687 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5012 , i =  21
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([6269, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 18875 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5012 , i =  22
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([7137, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 18007 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5012 , i =  23
!!! Drift in window 1 (p-value: 0.000000, severity: 0.6546)
Drift detected, update the model...
Selected representative old samples: torch.Size([9315, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 15829 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.1727, epochs=23, severity=0.6546
epoch =  0
seed =  5012 , i =  24
Selected representative old samples: torch.Size([6995, 121])
Selected representative new samples: torch.Size([3323, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[6258 3452]
 [6032 6567]]
Accuracy  0.5748800932359137
Precision  0.6554546361912367
Recall  0.5212318437971267
F1 score  0.5806879476523124
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
[[ 8370  1341]
 [ 1731 11102]]
Accuracy  0.8637331440738112
Precision  0.8922285622438318
Recall  0.8651133795683005
F1 score  0.8784617819275201
seed =  5013 , i =  0
No drift in window 1 (p-value: 0.183196, severity: 0.0000)
Selected representative old samples: torch.Size([57, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5013 , i =  1
No drift in window 1 (p-value: 0.490874, severity: 0.0000)
Selected representative old samples: torch.Size([52, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5013 , i =  2
No drift in window 1 (p-value: 0.998727, severity: 0.0000)
Selected representative old samples: torch.Size([124, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5013 , i =  3
No drift in window 1 (p-value: 0.634081, severity: 0.0000)
Selected representative old samples: torch.Size([165, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5013 , i =  4
No drift in window 1 (p-value: 0.809396, severity: 0.0000)
Selected representative old samples: torch.Size([231, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5013 , i =  5
No drift in window 1 (p-value: 0.544810, severity: 0.0000)
Selected representative old samples: torch.Size([185, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5013 , i =  6
No drift in window 1 (p-value: 0.438510, severity: 0.0000)
Selected representative old samples: torch.Size([262, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5013 , i =  7
No drift in window 1 (p-value: 0.995311, severity: 0.0000)
Selected representative old samples: torch.Size([188, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5013 , i =  8
No drift in window 1 (p-value: 0.951797, severity: 0.0000)
Selected representative old samples: torch.Size([220, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5013 , i =  9
No drift in window 1 (p-value: 0.848085, severity: 0.0000)
Selected representative old samples: torch.Size([236, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5013 , i =  10
No drift in window 1 (p-value: 0.620490, severity: 0.0000)
Selected representative old samples: torch.Size([365, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5013 , i =  11
No drift in window 1 (p-value: 0.095568, severity: 0.0000)
Selected representative old samples: torch.Size([251, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5013 , i =  12
No drift in window 1 (p-value: 0.381936, severity: 0.0000)
Selected representative old samples: torch.Size([76, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5013 , i =  13
No drift in window 1 (p-value: 0.873630, severity: 0.0000)
Selected representative old samples: torch.Size([136, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5013 , i =  14
!!! Drift in window 1 (p-value: 0.003258, severity: 0.2487)
Drift detected, update the model...
Selected representative old samples: torch.Size([225, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 24919 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.3756, epochs=14, severity=0.2487
epoch =  0
seed =  5013 , i =  15
!!! Drift in window 1 (p-value: 0.010967, severity: 0.1960)
Drift detected, update the model...
Selected representative old samples: torch.Size([4799, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 20345 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.4020, epochs=13, severity=0.1960
epoch =  0
seed =  5013 , i =  16
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([6863, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 18281 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5013 , i =  17
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([7060, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 18084 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5013 , i =  18
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([9784, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 15360 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5013 , i =  19
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([3174, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 21970 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5013 , i =  20
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([9408, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 15736 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5013 , i =  21
!!! Drift in window 1 (p-value: 0.000018, severity: 0.4743)
Drift detected, update the model...
Selected representative old samples: torch.Size([676, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 24468 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.2629, epochs=19, severity=0.4743
epoch =  0
seed =  5013 , i =  22
!!! Drift in window 1 (p-value: 0.000000, severity: 0.7931)
Drift detected, update the model...
Selected representative old samples: torch.Size([88, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 25056 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.1034, epochs=25, severity=0.7931
epoch =  0
seed =  5013 , i =  23
No drift in window 1 (p-value: 0.999970, severity: 0.0000)
Selected representative old samples: torch.Size([0, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5013 , i =  24
Selected representative old samples: torch.Size([979, 121])
Selected representative new samples: torch.Size([3323, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[9510  176]
 [9531 3095]]
Accuracy  0.5649426317676587
Precision  0.9461938245184959
Recall  0.24512909868525265
F1 score  0.38938164433540917
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
[[ 8534  1177]
 [ 2139 10694]]
Accuracy  0.8529098651525905
Precision  0.9008508129053997
Recall  0.8333203459830125
F1 score  0.865770725388601
seed =  5014 , i =  0
No drift in window 1 (p-value: 0.999981, severity: 0.0000)
Selected representative old samples: torch.Size([172, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  1
No drift in window 1 (p-value: 0.400700, severity: 0.0000)
Selected representative old samples: torch.Size([109, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  2
No drift in window 1 (p-value: 0.431381, severity: 0.0000)
Selected representative old samples: torch.Size([109, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  3
No drift in window 1 (p-value: 0.999875, severity: 0.0000)
Selected representative old samples: torch.Size([92, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  4
No drift in window 1 (p-value: 0.706364, severity: 0.0000)
Selected representative old samples: torch.Size([142, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  5
No drift in window 1 (p-value: 0.998762, severity: 0.0000)
Selected representative old samples: torch.Size([186, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  6
No drift in window 1 (p-value: 0.905748, severity: 0.0000)
Selected representative old samples: torch.Size([191, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  7
No drift in window 1 (p-value: 0.724609, severity: 0.0000)
Selected representative old samples: torch.Size([116, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  8
No drift in window 1 (p-value: 0.874517, severity: 0.0000)
Selected representative old samples: torch.Size([222, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  9
No drift in window 1 (p-value: 0.988669, severity: 0.0000)
Selected representative old samples: torch.Size([120, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  10
No drift in window 1 (p-value: 0.948904, severity: 0.0000)
Selected representative old samples: torch.Size([236, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  11
No drift in window 1 (p-value: 0.997217, severity: 0.0000)
Selected representative old samples: torch.Size([99, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  12
No drift in window 1 (p-value: 0.227447, severity: 0.0000)
Selected representative old samples: torch.Size([89, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  13
No drift in window 1 (p-value: 0.681639, severity: 0.0000)
Selected representative old samples: torch.Size([155, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  14
No drift in window 1 (p-value: 0.747481, severity: 0.0000)
Selected representative old samples: torch.Size([137, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  15
No drift in window 1 (p-value: 0.746837, severity: 0.0000)
Selected representative old samples: torch.Size([141, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  16
No drift in window 1 (p-value: 0.110697, severity: 0.0000)
Selected representative old samples: torch.Size([255, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  17
No drift in window 1 (p-value: 0.867213, severity: 0.0000)
Selected representative old samples: torch.Size([168, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  18
No drift in window 1 (p-value: 0.343564, severity: 0.0000)
Selected representative old samples: torch.Size([320, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  19
No drift in window 1 (p-value: 0.097349, severity: 0.0000)
Selected representative old samples: torch.Size([78, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  20
!!! Drift in window 1 (p-value: 0.001287, severity: 0.2891)
Drift detected, update the model...
Selected representative old samples: torch.Size([289, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 24855 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.3555, epochs=15, severity=0.2891
epoch =  0
seed =  5014 , i =  21
No drift in window 1 (p-value: 0.321621, severity: 0.0000)
Selected representative old samples: torch.Size([4864, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5014 , i =  22
!!! Drift in window 1 (p-value: 0.026187, severity: 0.1582)
Drift detected, update the model...
Selected representative old samples: torch.Size([600, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 24544 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.4209, epochs=13, severity=0.1582
epoch =  0
seed =  5014 , i =  23
!!! Drift in window 1 (p-value: 0.007990, severity: 0.2097)
Drift detected, update the model...
Selected representative old samples: torch.Size([3440, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 21704 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.3951, epochs=14, severity=0.2097
epoch =  0
seed =  5014 , i =  24
Selected representative old samples: torch.Size([6490, 121])
Selected representative new samples: torch.Size([3323, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[8924  786]
 [4942 7652]]
Accuracy  0.7431850789096126
Precision  0.9068499644465513
Recall  0.6075909163093537
F1 score  0.7276531000380373
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
[[ 8469  1242]
 [ 2703 10130]]
Accuracy  0.8250088715400994
Precision  0.890784382694337
Recall  0.7893711524974675
F1 score  0.8370171452179302
seed =  5015 , i =  0
No drift in window 1 (p-value: 0.414203, severity: 0.0000)
Selected representative old samples: torch.Size([232, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5015 , i =  1
!!! Drift in window 1 (p-value: 0.031527, severity: 0.1501)
Drift detected, update the model...
Selected representative old samples: torch.Size([160, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 24984 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.4249, epochs=13, severity=0.1501
epoch =  0
seed =  5015 , i =  2
!!! Drift in window 1 (p-value: 0.004017, severity: 0.2396)
Drift detected, update the model...
Selected representative old samples: torch.Size([5044, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 20100 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.3802, epochs=14, severity=0.2396
epoch =  0
seed =  5015 , i =  3
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([6838, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 18306 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5015 , i =  4
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([6650, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 18494 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5015 , i =  5
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([8142, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 17002 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5015 , i =  6
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([4198, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 20946 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5015 , i =  7
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([4808, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 20336 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5015 , i =  8
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([6898, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 18246 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5015 , i =  9
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([6518, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 18626 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5015 , i =  10
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([0, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative samples selected (0). Selecting additional random samples.
Buffer memory has extra space for 25144 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=30, severity=1.0000
epoch =  0
seed =  5015 , i =  11
No drift in window 1 (p-value: 0.200289, severity: 0.0000)
Selected representative old samples: torch.Size([1944, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5015 , i =  12
No drift in window 1 (p-value: 0.553149, severity: 0.0000)
Selected representative old samples: torch.Size([1934, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5015 , i =  13
No drift in window 1 (p-value: 0.135394, severity: 0.0000)
Selected representative old samples: torch.Size([0, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative new samples selected (0). Selecting additional random samples.
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5015 , i =  14
No drift in window 1 (p-value: 0.903651, severity: 0.0000)
Selected representative old samples: torch.Size([7, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5015 , i =  15
No drift in window 1 (p-value: 0.758169, severity: 0.0000)
Selected representative old samples: torch.Size([6, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5015 , i =  16
No drift in window 1 (p-value: 0.907967, severity: 0.0000)
Selected representative old samples: torch.Size([1896, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5015 , i =  17
No drift in window 1 (p-value: 0.122180, severity: 0.0000)
Selected representative old samples: torch.Size([1889, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5015 , i =  18
No drift in window 1 (p-value: 0.864508, severity: 0.0000)
Selected representative old samples: torch.Size([14, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5015 , i =  19
No drift in window 1 (p-value: 0.784198, severity: 0.0000)
Selected representative old samples: torch.Size([9, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5015 , i =  20
!!! Drift in window 1 (p-value: 0.002211, severity: 0.2656)
Drift detected, update the model...
Selected representative old samples: torch.Size([1876, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 23268 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.3672, epochs=15, severity=0.2656
epoch =  0
seed =  5015 , i =  21
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([5656, 121])
Selected representative new samples: torch.Size([5000, 121])
Buffer memory has extra space for 19488 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=29, severity=1.0000
epoch =  0
seed =  5015 , i =  22
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([0, 121])
Selected representative new samples: torch.Size([0, 121])
Not enough representative samples selected (0). Selecting additional random samples.
Buffer memory has extra space for 25144 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=29, severity=1.0000
epoch =  0
seed =  5015 , i =  23
No drift in window 1 (p-value: 0.686521, severity: 0.0000)
Selected representative old samples: torch.Size([33, 121])
Selected representative new samples: torch.Size([5000, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
seed =  5015 , i =  24
Selected representative old samples: torch.Size([57, 121])
Selected representative new samples: torch.Size([3323, 121])
Training with adaptive_lwf=0.5000, epochs=10, severity=0.0000
epoch =  0
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[8460 1229]
 [2965 9647]]
Accuracy  0.811936684453612
Precision  0.8869988966531813
Recall  0.7649064383127181
F1 score  0.8214407356948229

# 2.unsw
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5012 , i =  4
No drift in window 1 (p-value: 0.071573, severity: 0.0000)
Selected representative old samples: torch.Size([20075, 196])
Selected representative new samples: torch.Size([20000, 196])
Not enough non-representative old samples to remove (125). Removing additional representative samples.
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5012 , i =  5
!!! Drift in window 1 (p-value: 0.012909, severity: 0.1889)
Drift detected, update the model...
Selected representative old samples: torch.Size([0, 196])
Selected representative new samples: torch.Size([0, 196])
Not enough representative samples selected (0). Selecting additional random samples.
Buffer memory has extra space for 34868 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.4055, epochs=124, severity=0.1889
epoch =  0
epoch =  50
epoch =  100
seed =  5012 , i =  6
No drift in window 1 (p-value: 0.470752, severity: 0.0000)
Selected representative old samples: torch.Size([5151, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5012 , i =  7
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([5148, 196])
Selected representative new samples: torch.Size([20000, 196])
Buffer memory has extra space for 29720 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=270, severity=1.0000
epoch =  0
epoch =  50
epoch =  100
epoch =  150
epoch =  200
epoch =  250
seed =  5012 , i =  8
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([34941, 196])
Selected representative new samples: torch.Size([20000, 196])
Not enough non-representative old samples to remove (127). Removing additional representative samples.
Training with adaptive_lwf=0.0000, epochs=270, severity=1.0000
epoch =  0
epoch =  50
epoch =  100
epoch =  150
epoch =  200
epoch =  250
seed =  5012 , i =  9
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([0, 196])
Selected representative new samples: torch.Size([0, 196])
Not enough representative samples selected (0). Selecting additional random samples.
Buffer memory has extra space for 34868 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=270, severity=1.0000
epoch =  0
epoch =  50
epoch =  100
epoch =  150
epoch =  200
epoch =  250
seed =  5012 , i =  10
No drift in window 1 (p-value: 0.100940, severity: 0.0000)
Selected representative old samples: torch.Size([5139, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5012 , i =  11
Selected representative old samples: torch.Size([14713, 196])
Selected representative new samples: torch.Size([2605, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[21495 15209]
 [   81 44549]]
Accuracy  0.8120097376251014
Precision  0.7454901435791024
Recall  0.998185077302263
F1 score  0.853527225351573
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
[[22458 14542]
 [  482 44850]]
Accuracy  0.8175193120536365
Precision  0.7551522090517241
Recall  0.9893673343333628
F1 score  0.8565371834536496
seed =  5013 , i =  0
No drift in window 1 (p-value: 0.833235, severity: 0.0000)
Selected representative old samples: torch.Size([256, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5013 , i =  1
No drift in window 1 (p-value: 0.772182, severity: 0.0000)
Selected representative old samples: torch.Size([29, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5013 , i =  2
No drift in window 1 (p-value: 0.188732, severity: 0.0000)
Selected representative old samples: torch.Size([2902, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5013 , i =  3
No drift in window 1 (p-value: 0.056708, severity: 0.0000)
Selected representative old samples: torch.Size([9450, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5013 , i =  4
No drift in window 1 (p-value: 0.050074, severity: 0.0000)
Selected representative old samples: torch.Size([9165, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5013 , i =  5
!!! Drift in window 1 (p-value: 0.001720, severity: 0.2764)
Drift detected, update the model...
Selected representative old samples: torch.Size([29092, 196])
Selected representative new samples: torch.Size([20000, 196])
Buffer memory has extra space for 5776 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.3618, epochs=139, severity=0.2764
epoch =  0
epoch =  50
epoch =  100
seed =  5013 , i =  6
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([0, 196])
Selected representative new samples: torch.Size([0, 196])
Not enough representative samples selected (0). Selecting additional random samples.
Buffer memory has extra space for 34868 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=270, severity=1.0000
epoch =  0
epoch =  50
epoch =  100
epoch =  150
epoch =  200
epoch =  250
seed =  5013 , i =  7
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([13951, 196])
Selected representative new samples: torch.Size([20000, 196])
Buffer memory has extra space for 20917 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=270, severity=1.0000
epoch =  0
epoch =  50
epoch =  100
epoch =  150
epoch =  200
epoch =  250
seed =  5013 , i =  8
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([29181, 196])
Selected representative new samples: torch.Size([20000, 196])
Buffer memory has extra space for 5687 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=270, severity=1.0000
epoch =  0
epoch =  50
epoch =  100
epoch =  150
epoch =  200
epoch =  250
seed =  5013 , i =  9
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([0, 196])
Selected representative new samples: torch.Size([0, 196])
Not enough representative samples selected (0). Selecting additional random samples.
Buffer memory has extra space for 34868 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=270, severity=1.0000
epoch =  0
epoch =  50
epoch =  100
epoch =  150
epoch =  200
epoch =  250
seed =  5013 , i =  10
No drift in window 1 (p-value: 0.753691, severity: 0.0000)
Selected representative old samples: torch.Size([9790, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5013 , i =  11
Selected representative old samples: torch.Size([19928, 196])
Selected representative new samples: torch.Size([2605, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[21334 15415]
 [  107 44482]]
Accuracy  0.8091666871572942
Precision  0.7426415346344558
Recall  0.9976003050079616
F1 score  0.8514442126217866
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
[[22165 14835]
 [  294 45038]]
Accuracy  0.8162439877568868
Precision  0.7522255440682779
Recall  0.993514515132798
F1 score  0.8561950477638895
seed =  5014 , i =  0
!!! Drift in window 1 (p-value: 0.005214, severity: 0.2283)
Drift detected, update the model...
Selected representative old samples: torch.Size([17328, 196])
Selected representative new samples: torch.Size([20000, 196])
Buffer memory has extra space for 17540 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.3859, epochs=131, severity=0.2283
epoch =  0
epoch =  50
epoch =  100
seed =  5014 , i =  1
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([31618, 196])
Selected representative new samples: torch.Size([20000, 196])
Buffer memory has extra space for 3250 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=270, severity=1.0000
epoch =  0
epoch =  50
epoch =  100
epoch =  150
epoch =  200
epoch =  250
seed =  5014 , i =  2
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([0, 196])
Selected representative new samples: torch.Size([0, 196])
Not enough representative samples selected (0). Selecting additional random samples.
Buffer memory has extra space for 34868 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=270, severity=1.0000
epoch =  0
epoch =  50
epoch =  100
epoch =  150
epoch =  200
epoch =  250
seed =  5014 , i =  3
No drift in window 1 (p-value: 0.819171, severity: 0.0000)
Selected representative old samples: torch.Size([19882, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5014 , i =  4
!!! Drift in window 1 (p-value: 0.003164, severity: 0.2500)
Drift detected, update the model...
Selected representative old samples: torch.Size([20106, 196])
Selected representative new samples: torch.Size([20000, 196])
Not enough non-representative old samples to remove (94). Removing additional representative samples.
Buffer memory has extra space for 14868 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.3750, epochs=134, severity=0.2500
epoch =  0
epoch =  50
epoch =  100
seed =  5014 , i =  5
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([0, 196])
Selected representative new samples: torch.Size([0, 196])
Not enough representative samples selected (0). Selecting additional random samples.
Buffer memory has extra space for 34868 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=270, severity=1.0000
epoch =  0
epoch =  50
epoch =  100
epoch =  150
epoch =  200
epoch =  250
seed =  5014 , i =  6
No drift in window 1 (p-value: 0.837348, severity: 0.0000)
Selected representative old samples: torch.Size([5424, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5014 , i =  7
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([5397, 196])
Selected representative new samples: torch.Size([20000, 196])
Buffer memory has extra space for 29471 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=270, severity=1.0000
epoch =  0
epoch =  50
epoch =  100
epoch =  150
epoch =  200
epoch =  250
seed =  5014 , i =  8
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([12423, 196])
Selected representative new samples: torch.Size([20000, 196])
Buffer memory has extra space for 22445 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=270, severity=1.0000
epoch =  0
epoch =  50
epoch =  100
epoch =  150
epoch =  200
epoch =  250
seed =  5014 , i =  9
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([45, 196])
Selected representative new samples: torch.Size([20000, 196])
Buffer memory has extra space for 34823 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=270, severity=1.0000
epoch =  0
epoch =  50
epoch =  100
epoch =  150
epoch =  200
epoch =  250
seed =  5014 , i =  10
No drift in window 1 (p-value: 0.507784, severity: 0.0000)
Selected representative old samples: torch.Size([0, 196])
Selected representative new samples: torch.Size([0, 196])
Not enough representative new samples selected (0). Selecting additional random samples.
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5014 , i =  11
Selected representative old samples: torch.Size([16679, 196])
Selected representative new samples: torch.Size([2605, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[19786 17126]
 [  204 44223]]
Accuracy  0.7869410737776467
Precision  0.7208430455264144
Recall  0.9954081977176041
F1 score  0.8361632128271063
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
[[22434 14566]
 [  427 44905]]
Accuracy  0.8178958363698198
Precision  0.7550739015654688
Recall  0.9905806053119209
F1 score  0.8569411180977642
seed =  5015 , i =  0
No drift in window 1 (p-value: 0.899179, severity: 0.0000)
Selected representative old samples: torch.Size([19856, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5015 , i =  1
No drift in window 1 (p-value: 0.298957, severity: 0.0000)
Selected representative old samples: torch.Size([0, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5015 , i =  2
!!! Drift in window 1 (p-value: 0.023585, severity: 0.1627)
Drift detected, update the model...
Selected representative old samples: torch.Size([0, 196])
Selected representative new samples: torch.Size([20000, 196])
Buffer memory has extra space for 34868 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.4186, epochs=119, severity=0.1627
epoch =  0
epoch =  50
epoch =  100
seed =  5015 , i =  3
No drift in window 1 (p-value: 0.064803, severity: 0.0000)
Selected representative old samples: torch.Size([30592, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5015 , i =  4
No drift in window 1 (p-value: 0.574145, severity: 0.0000)
Selected representative old samples: torch.Size([9803, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5015 , i =  5
No drift in window 1 (p-value: 0.475751, severity: 0.0000)
Selected representative old samples: torch.Size([13926, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5015 , i =  6
No drift in window 1 (p-value: 0.698731, severity: 0.0000)
Selected representative old samples: torch.Size([311, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5015 , i =  7
!!! Drift in window 1 (p-value: 0.000000, severity: 1.0000)
Drift detected, update the model...
Selected representative old samples: torch.Size([226, 196])
Selected representative new samples: torch.Size([20000, 196])
Buffer memory has extra space for 34642 samples. Adding new samples with pseudo labels.
Training with adaptive_lwf=0.0000, epochs=270, severity=1.0000
epoch =  0
epoch =  50
epoch =  100
epoch =  150
epoch =  200
epoch =  250
seed =  5015 , i =  8
No drift in window 1 (p-value: 0.453457, severity: 0.0000)
Selected representative old samples: torch.Size([212, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5015 , i =  9
No drift in window 1 (p-value: 0.304165, severity: 0.0000)
Selected representative old samples: torch.Size([307, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5015 , i =  10
No drift in window 1 (p-value: 0.311434, severity: 0.0000)
Selected representative old samples: torch.Size([176, 196])
Selected representative new samples: torch.Size([20000, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
seed =  5015 , i =  11
Selected representative old samples: torch.Size([138, 196])
Selected representative new samples: torch.Size([2605, 196])
Training with adaptive_lwf=0.5000, epochs=90, severity=0.0000
epoch =  0
epoch =  50
--------------------------- performance_after_continual_training -----------------------------------
Confusion matrix
[[22918 14078]
 [  640 43703]]
Accuracy  0.8190535905285288
Precision  0.7563558955365951
Recall  0.9855670568071624
F1 score  0.8558810857389056