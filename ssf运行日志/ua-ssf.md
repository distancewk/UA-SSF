# 1.nsl
(ssf) jackson@omnisky:~/wkcodenew/SSF-Strategic-Selection-and-Forgetting-main$ python ssf.py --mode ua-ssf --dataset nsl --epochs 200 --epoch_1 20 --sample_interval 5000 --num_labeled_sample 50 --opt_old_lr 100 --opt_new_lr 8 --new_sample_weight 3 --cuda 0
Random seed set to: 5011

============================================================
Seed 5011 (1/5) | dataset=nsl | mode=ua-ssf
============================================================
[Config] train=125973, test=22544, memory=25194, input_dim=121
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8793  Pre=0.8862  Rec=0.9040  F1=0.8950
  Confusion: TP=11601 FP=1490 FN=1232 TN=8221
  Samples: old=118, new=5000[W00] stable(p=0.9730) | lwf=0.50 ep=10 | buf=25194
  Samples: old=88, new=5000[W01] stable(p=0.5374) | lwf=0.50 ep=10 | buf=25194
  Samples: old=401, new=5000[W02] stable(p=0.6132) | lwf=0.50 ep=10 | buf=25194
  Samples: old=70, new=5000[W03] stable(p=0.5768) | lwf=0.50 ep=10 | buf=25194
  Samples: old=229, new=5000[W04] stable(p=0.5645) | lwf=0.50 ep=10 | buf=25194
  Samples: old=168, new=5000 | pseudo_fill=24976[W05] DRIFT(sev=0.18) | lwf=0.41 ep=13 | buf=10168
  Samples: old=5182, new=5000[W06] stable(p=0.0683) | lwf=0.50 ep=10 | buf=10168
  Samples: old=4711, new=5000 | pseudo_fill=20433[W07] DRIFT(sev=0.16) | lwf=0.42 ep=13 | buf=14711
  Samples: old=6220, new=5000 | pseudo_fill=18924[W08] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=16220
  Samples: old=6374, new=5000 | pseudo_fill=18770[W09] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=16374
  Samples: old=5384, new=5000 | pseudo_fill=19760[W10] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=15384
  Samples: old=0, new=0 | new补充=50 | pseudo_fill=25144[W11] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=5050
  Samples: old=2178, new=5000[W12] stable(p=0.2017) | lwf=0.50 ep=10 | buf=5050
  Samples: old=2165, new=5000 | pseudo_fill=22979[W13] DRIFT(sev=0.19) | lwf=0.40 ep=13 | buf=12165
  Samples: old=5490, new=5000 | pseudo_fill=19654[W14] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=15490
  Samples: old=7304, new=5000 | pseudo_fill=17840[W15] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=17304
  Samples: old=6847, new=5000 | pseudo_fill=18297[W16] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=16847
  Samples: old=8227, new=5000 | pseudo_fill=16917[W17] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=18227
  Samples: old=4736, new=5000 | pseudo_fill=20408[W18] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=14736
  Samples: old=5200, new=5000 | pseudo_fill=19944[W19] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=15200
  Samples: old=4626, new=5000 | pseudo_fill=20518[W20] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=14626
  Samples: old=7508, new=5000 | pseudo_fill=17636[W21] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=17508
  Samples: old=0, new=0 | new补充=50 | pseudo_fill=25144[W22] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=5050
  Samples: old=30, new=5000[W23] stable(p=0.5641) | lwf=0.50 ep=10 | buf=5050
  Samples: old=67, new=3323[W24] stable(p=1.0000) | lwf=0.50 ep=10 | buf=5050
[After CL]
  Acc=0.7252  Pre=0.8231  Rec=0.6563  F1=0.7303
  Confusion: TP=8301 FP=1784 FN=4348 TN=7884
Random seed set to: 5012

============================================================
Seed 5012 (2/5) | dataset=nsl | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8256  Pre=0.9108  Rec=0.7690  F1=0.8339
  Confusion: TP=9869 FP=967 FN=2964 TN=8744
  Samples: old=371, new=5000[W00] stable(p=0.3017) | lwf=0.50 ep=10 | buf=25194
  Samples: old=117, new=5000[W01] stable(p=0.7111) | lwf=0.50 ep=10 | buf=25194
  Samples: old=96, new=5000[W02] stable(p=0.7018) | lwf=0.50 ep=10 | buf=25194
  Samples: old=315, new=5000[W03] stable(p=0.6288) | lwf=0.50 ep=10 | buf=25194
  Samples: old=82, new=5000[W04] stable(p=0.5512) | lwf=0.50 ep=10 | buf=25194
  Samples: old=281, new=5000[W05] stable(p=0.3641) | lwf=0.50 ep=10 | buf=25194
  Samples: old=99, new=5000[W06] stable(p=0.9966) | lwf=0.50 ep=10 | buf=25194
  Samples: old=112, new=5000[W07] stable(p=0.4016) | lwf=0.50 ep=10 | buf=25194
  Samples: old=265, new=5000[W08] stable(p=0.7331) | lwf=0.50 ep=10 | buf=25194
  Samples: old=288, new=5000[W09] stable(p=0.6790) | lwf=0.50 ep=10 | buf=25194
  Samples: old=220, new=5000[W10] stable(p=0.1319) | lwf=0.50 ep=10 | buf=25194
  Samples: old=125, new=5000[W11] stable(p=0.7348) | lwf=0.50 ep=10 | buf=25194
  Samples: old=151, new=5000[W12] stable(p=0.0657) | lwf=0.50 ep=10 | buf=25194
  Samples: old=81, new=5000[W13] stable(p=0.5901) | lwf=0.50 ep=10 | buf=25194
  Samples: old=88, new=5000[W14] stable(p=0.1056) | lwf=0.50 ep=10 | buf=25194
  Samples: old=200, new=5000[W15] stable(p=0.3308) | lwf=0.50 ep=10 | buf=25194
  Samples: old=202, new=5000[W16] stable(p=0.4029) | lwf=0.50 ep=10 | buf=25194
  Samples: old=93, new=5000[W17] stable(p=0.1489) | lwf=0.50 ep=10 | buf=25194
  Samples: old=11648, new=5000 | pseudo_fill=13496[W18] DRIFT(sev=0.20) | lwf=0.40 ep=13 | buf=21648
  Samples: old=7716, new=5000 | pseudo_fill=17428[W19] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=17716
  Samples: old=7457, new=5000 | pseudo_fill=17687[W20] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=17457
  Samples: old=6269, new=5000 | pseudo_fill=18875[W21] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=16269
  Samples: old=7137, new=5000 | pseudo_fill=18007[W22] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=17137
  Samples: old=9315, new=5000 | pseudo_fill=15829[W23] DRIFT(sev=0.65) | lwf=0.17 ep=23 | buf=19315
  Samples: old=6995, new=3323[W24] stable(p=1.0000) | lwf=0.50 ep=10 | buf=19315
[After CL]
  Acc=0.5749  Pre=0.6555  Rec=0.5212  F1=0.5807
  Confusion: TP=6567 FP=3452 FN=6032 TN=6258
Random seed set to: 5013

============================================================
Seed 5013 (3/5) | dataset=nsl | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8637  Pre=0.8922  Rec=0.8651  F1=0.8785
  Confusion: TP=11102 FP=1341 FN=1731 TN=8370
  Samples: old=57, new=5000[W00] stable(p=0.1832) | lwf=0.50 ep=10 | buf=25194
  Samples: old=52, new=5000[W01] stable(p=0.4909) | lwf=0.50 ep=10 | buf=25194
  Samples: old=124, new=5000[W02] stable(p=0.9987) | lwf=0.50 ep=10 | buf=25194
  Samples: old=165, new=5000[W03] stable(p=0.6341) | lwf=0.50 ep=10 | buf=25194
  Samples: old=231, new=5000[W04] stable(p=0.8094) | lwf=0.50 ep=10 | buf=25194
  Samples: old=185, new=5000[W05] stable(p=0.5448) | lwf=0.50 ep=10 | buf=25194
  Samples: old=262, new=5000[W06] stable(p=0.4385) | lwf=0.50 ep=10 | buf=25194
  Samples: old=188, new=5000[W07] stable(p=0.9953) | lwf=0.50 ep=10 | buf=25194
  Samples: old=220, new=5000[W08] stable(p=0.9518) | lwf=0.50 ep=10 | buf=25194
  Samples: old=236, new=5000[W09] stable(p=0.8481) | lwf=0.50 ep=10 | buf=25194
  Samples: old=365, new=5000[W10] stable(p=0.6205) | lwf=0.50 ep=10 | buf=25194
  Samples: old=251, new=5000[W11] stable(p=0.0956) | lwf=0.50 ep=10 | buf=25194
  Samples: old=76, new=5000[W12] stable(p=0.3819) | lwf=0.50 ep=10 | buf=25194
  Samples: old=136, new=5000[W13] stable(p=0.8736) | lwf=0.50 ep=10 | buf=25194
  Samples: old=225, new=5000 | pseudo_fill=24919[W14] DRIFT(sev=0.25) | lwf=0.38 ep=14 | buf=10225
  Samples: old=4799, new=5000 | pseudo_fill=20345[W15] DRIFT(sev=0.20) | lwf=0.40 ep=13 | buf=14799
  Samples: old=6863, new=5000 | pseudo_fill=18281[W16] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=16863
  Samples: old=7060, new=5000 | pseudo_fill=18084[W17] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=17060
  Samples: old=9784, new=5000 | pseudo_fill=15360[W18] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=19784
  Samples: old=3174, new=5000 | pseudo_fill=21970[W19] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=13174
  Samples: old=9408, new=5000 | pseudo_fill=15736[W20] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=19408
  Samples: old=676, new=5000 | pseudo_fill=24468[W21] DRIFT(sev=0.47) | lwf=0.26 ep=19 | buf=10676
  Samples: old=88, new=5000 | pseudo_fill=25056[W22] DRIFT(sev=0.79) | lwf=0.10 ep=25 | buf=10088
  Samples: old=0, new=0 | new补充=50[W23] stable(p=1.0000) | lwf=0.50 ep=10 | buf=10088
  Samples: old=979, new=3323[W24] stable(p=1.0000) | lwf=0.50 ep=10 | buf=10088
[After CL]
  Acc=0.5649  Pre=0.9462  Rec=0.2451  F1=0.3894
  Confusion: TP=3095 FP=176 FN=9531 TN=9510
Random seed set to: 5014

============================================================
Seed 5014 (4/5) | dataset=nsl | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8529  Pre=0.9009  Rec=0.8333  F1=0.8658
  Confusion: TP=10694 FP=1177 FN=2139 TN=8534
  Samples: old=172, new=5000[W00] stable(p=1.0000) | lwf=0.50 ep=10 | buf=25194
  Samples: old=109, new=5000[W01] stable(p=0.4007) | lwf=0.50 ep=10 | buf=25194
  Samples: old=109, new=5000[W02] stable(p=0.4314) | lwf=0.50 ep=10 | buf=25194
  Samples: old=92, new=5000[W03] stable(p=0.9999) | lwf=0.50 ep=10 | buf=25194
  Samples: old=142, new=5000[W04] stable(p=0.7064) | lwf=0.50 ep=10 | buf=25194
  Samples: old=186, new=5000[W05] stable(p=0.9988) | lwf=0.50 ep=10 | buf=25194
  Samples: old=191, new=5000[W06] stable(p=0.9057) | lwf=0.50 ep=10 | buf=25194
  Samples: old=116, new=5000[W07] stable(p=0.7246) | lwf=0.50 ep=10 | buf=25194
  Samples: old=222, new=5000[W08] stable(p=0.8745) | lwf=0.50 ep=10 | buf=25194
  Samples: old=120, new=5000[W09] stable(p=0.9887) | lwf=0.50 ep=10 | buf=25194
  Samples: old=236, new=5000[W10] stable(p=0.9489) | lwf=0.50 ep=10 | buf=25194
  Samples: old=99, new=5000[W11] stable(p=0.9972) | lwf=0.50 ep=10 | buf=25194
  Samples: old=89, new=5000[W12] stable(p=0.2274) | lwf=0.50 ep=10 | buf=25194
  Samples: old=155, new=5000[W13] stable(p=0.6816) | lwf=0.50 ep=10 | buf=25194
  Samples: old=137, new=5000[W14] stable(p=0.7475) | lwf=0.50 ep=10 | buf=25194
  Samples: old=141, new=5000[W15] stable(p=0.7468) | lwf=0.50 ep=10 | buf=25194
  Samples: old=255, new=5000[W16] stable(p=0.1107) | lwf=0.50 ep=10 | buf=25194
  Samples: old=168, new=5000[W17] stable(p=0.8672) | lwf=0.50 ep=10 | buf=25194
  Samples: old=320, new=5000[W18] stable(p=0.3436) | lwf=0.50 ep=10 | buf=25194
  Samples: old=78, new=5000[W19] stable(p=0.0973) | lwf=0.50 ep=10 | buf=25194
  Samples: old=289, new=5000 | pseudo_fill=24855[W20] DRIFT(sev=0.29) | lwf=0.36 ep=15 | buf=10289
  Samples: old=4864, new=5000[W21] stable(p=0.3216) | lwf=0.50 ep=10 | buf=10289
  Samples: old=600, new=5000 | pseudo_fill=24544[W22] DRIFT(sev=0.16) | lwf=0.42 ep=13 | buf=10600
  Samples: old=3440, new=5000 | pseudo_fill=21704[W23] DRIFT(sev=0.21) | lwf=0.40 ep=14 | buf=13440
  Samples: old=6490, new=3323[W24] stable(p=1.0000) | lwf=0.50 ep=10 | buf=13440
[After CL]
  Acc=0.7432  Pre=0.9068  Rec=0.6076  F1=0.7277
  Confusion: TP=7652 FP=786 FN=4942 TN=8924
Random seed set to: 5015

============================================================
Seed 5015 (5/5) | dataset=nsl | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8250  Pre=0.8908  Rec=0.7894  F1=0.8370
  Confusion: TP=10130 FP=1242 FN=2703 TN=8469
  Samples: old=232, new=5000[W00] stable(p=0.4142) | lwf=0.50 ep=10 | buf=25194
  Samples: old=160, new=5000 | pseudo_fill=24984[W01] DRIFT(sev=0.15) | lwf=0.42 ep=13 | buf=10160
  Samples: old=5044, new=5000 | pseudo_fill=20100[W02] DRIFT(sev=0.24) | lwf=0.38 ep=14 | buf=15044
  Samples: old=6838, new=5000 | pseudo_fill=18306[W03] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=16838
  Samples: old=6650, new=5000 | pseudo_fill=18494[W04] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=16650
  Samples: old=8142, new=5000 | pseudo_fill=17002[W05] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=18142
  Samples: old=4198, new=5000 | pseudo_fill=20946[W06] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=14198
  Samples: old=4808, new=5000 | pseudo_fill=20336[W07] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=14808
  Samples: old=6898, new=5000 | pseudo_fill=18246[W08] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=16898
  Samples: old=6518, new=5000 | pseudo_fill=18626[W09] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=16518
  Samples: old=0, new=0 | new补充=50 | pseudo_fill=25144[W10] DRIFT(sev=1.00) | lwf=0.00 ep=30 | buf=5050
  Samples: old=1944, new=5000[W11] stable(p=0.2003) | lwf=0.50 ep=10 | buf=5050
  Samples: old=1934, new=5000[W12] stable(p=0.5531) | lwf=0.50 ep=10 | buf=5050
  Samples: old=0, new=0 | new补充=50[W13] stable(p=0.1354) | lwf=0.50 ep=10 | buf=5050
  Samples: old=7, new=5000[W14] stable(p=0.9037) | lwf=0.50 ep=10 | buf=5050
  Samples: old=6, new=5000[W15] stable(p=0.7582) | lwf=0.50 ep=10 | buf=5050
  Samples: old=1896, new=5000[W16] stable(p=0.9080) | lwf=0.50 ep=10 | buf=5050
  Samples: old=1889, new=5000[W17] stable(p=0.1222) | lwf=0.50 ep=10 | buf=5050
  Samples: old=14, new=5000[W18] stable(p=0.8645) | lwf=0.50 ep=10 | buf=5050
  Samples: old=9, new=5000[W19] stable(p=0.7842) | lwf=0.50 ep=10 | buf=5050
  Samples: old=1876, new=5000 | pseudo_fill=23268[W20] DRIFT(sev=0.27) | lwf=0.37 ep=15 | buf=11876
  Samples: old=5656, new=5000 | pseudo_fill=19488[W21] DRIFT(sev=1.00) | lwf=0.00 ep=29 | buf=15656
  Samples: old=0, new=0 | new补充=50 | pseudo_fill=25144[W22] DRIFT(sev=1.00) | lwf=0.00 ep=29 | buf=5050
  Samples: old=33, new=5000[W23] stable(p=0.6865) | lwf=0.50 ep=10 | buf=5050
  Samples: old=57, new=3323[W24] stable(p=1.0000) | lwf=0.50 ep=10 | buf=5050
[After CL]
  Acc=0.8119  Pre=0.8870  Rec=0.7649  F1=0.8214
  Confusion: TP=9647 FP=1229 FN=2965 TN=8460

# 2.unsw
(ssf) jackson@omnisky:~/wkcodenew/SSF-Strategic-Selection-and-Forgetting-main$ python ssf.py --mode ua-ssf --dataset unsw --epochs 200 --epoch_1 180 --sample_interval 20000 --num_labeled_sample 200 --opt_old_lr 24 --opt_new_lr 50 --new_sample_weight 60 --cuda 0
Random seed set to: 5011

============================================================
Seed 5011 (1/5) | dataset=unsw | mode=ua-ssf
============================================================
[Config] train=175341, test=82332, memory=35068, input_dim=196
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8185  Pre=0.7557  Rec=0.9907  F1=0.8574
  Confusion: TP=44909 FP=14519 FN=423 TN=22481
  Samples: old=17435, new=20000[W00] stable(p=0.1100) | lwf=0.50 ep=90 | buf=35068
  Samples: old=24874, new=20000[W01] stable(p=0.1370) | lwf=0.50 ep=90 | buf=35068
  Samples: old=0, new=20000 | pseudo_fill=34868[W02] DRIFT(sev=0.39) | lwf=0.31 ep=159 | buf=35068
  Samples: old=9901, new=20000[W03] stable(p=0.1584) | lwf=0.50 ep=90 | buf=35068
  Samples: old=15070, new=20000[W04] stable(p=0.2079) | lwf=0.50 ep=90 | buf=35068
  Samples: old=0, new=20000 | pseudo_fill=34868[W05] DRIFT(sev=0.33) | lwf=0.33 ep=150 | buf=35068
  Samples: old=244, new=20000 | pseudo_fill=34624[W06] DRIFT(sev=0.18) | lwf=0.41 ep=122 | buf=35068
  Samples: old=225, new=20000 | pseudo_fill=34643[W07] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=35068
  Samples: old=239, new=20000[W08] stable(p=0.6188) | lwf=0.50 ep=90 | buf=35068
  Samples: old=206, new=20000[W09] stable(p=0.3512) | lwf=0.50 ep=90 | buf=35068
  Samples: old=125, new=20000 | pseudo_fill=34743[W10] DRIFT(sev=0.13) | lwf=0.43 ep=113 | buf=35068
  Samples: old=158, new=2605[W11] stable(p=1.0000) | lwf=0.50 ep=90 | buf=35068
[After CL]
  Acc=0.8176  Pre=0.7531  Rec=0.9898  F1=0.8554
  Confusion: TP=43881 FP=14384 FN=454 TN=22615
Random seed set to: 5012

============================================================
Seed 5012 (2/5) | dataset=unsw | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8178  Pre=0.7554  Rec=0.9895  F1=0.8567
  Confusion: TP=44857 FP=14527 FN=475 TN=22473
  Samples: old=17257, new=20000[W00] stable(p=0.6201) | lwf=0.50 ep=90 | buf=35068
  Samples: old=27365, new=20000 | pseudo_fill=7503[W01] DRIFT(sev=0.14) | lwf=0.43 ep=114 | buf=35068
  Samples: old=0, new=0 | new补充=200 | pseudo_fill=34868[W02] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=20200
  Samples: old=5113, new=20000[W03] stable(p=0.2854) | lwf=0.50 ep=90 | buf=20200
  Samples: old=20075, new=20000 | WARN: not enough non-rep old to remove (125)[W04] stable(p=0.0716) | lwf=0.50 ep=90 | buf=20200
  Samples: old=0, new=0 | new补充=200 | pseudo_fill=34868[W05] DRIFT(sev=0.19) | lwf=0.41 ep=124 | buf=20200
  Samples: old=5151, new=20000[W06] stable(p=0.4708) | lwf=0.50 ep=90 | buf=20200
  Samples: old=5148, new=20000 | pseudo_fill=29720[W07] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=35068
  Samples: old=34941, new=20000 | WARN: not enough non-rep old (127)[W08] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=35068
  Samples: old=0, new=0 | new补充=200 | pseudo_fill=34868[W09] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=20200
  Samples: old=5139, new=20000[W10] stable(p=0.1009) | lwf=0.50 ep=90 | buf=20200
  Samples: old=14713, new=2605[W11] stable(p=1.0000) | lwf=0.50 ep=90 | buf=20200
[After CL]
  Acc=0.8120  Pre=0.7455  Rec=0.9982  F1=0.8535
  Confusion: TP=44549 FP=15209 FN=81 TN=21495
Random seed set to: 5013

============================================================
Seed 5013 (3/5) | dataset=unsw | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8175  Pre=0.7552  Rec=0.9894  F1=0.8565
  Confusion: TP=44850 FP=14542 FN=482 TN=22458
  Samples: old=256, new=20000[W00] stable(p=0.8332) | lwf=0.50 ep=90 | buf=35068
  Samples: old=29, new=20000[W01] stable(p=0.7722) | lwf=0.50 ep=90 | buf=35068
  Samples: old=2902, new=20000[W02] stable(p=0.1887) | lwf=0.50 ep=90 | buf=35068
  Samples: old=9450, new=20000[W03] stable(p=0.0567) | lwf=0.50 ep=90 | buf=35068
  Samples: old=9165, new=20000[W04] stable(p=0.0501) | lwf=0.50 ep=90 | buf=35068
  Samples: old=29092, new=20000 | pseudo_fill=5776[W05] DRIFT(sev=0.28) | lwf=0.36 ep=139 | buf=35068
  Samples: old=0, new=0 | new补充=200 | pseudo_fill=34868[W06] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=20200
  Samples: old=13951, new=20000 | pseudo_fill=20917[W07] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=35068
  Samples: old=29181, new=20000 | pseudo_fill=5687[W08] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=35068
  Samples: old=0, new=0 | new补充=200 | pseudo_fill=34868[W09] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=20200
  Samples: old=9790, new=20000[W10] stable(p=0.7537) | lwf=0.50 ep=90 | buf=20200
  Samples: old=19928, new=2605[W11] stable(p=1.0000) | lwf=0.50 ep=90 | buf=20200
[After CL]
  Acc=0.8092  Pre=0.7426  Rec=0.9976  F1=0.8514
  Confusion: TP=44482 FP=15415 FN=107 TN=21334
Random seed set to: 5014

============================================================
Seed 5014 (4/5) | dataset=unsw | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8162  Pre=0.7522  Rec=0.9935  F1=0.8562
  Confusion: TP=45038 FP=14835 FN=294 TN=22165
  Samples: old=17328, new=20000 | pseudo_fill=17540[W00] DRIFT(sev=0.23) | lwf=0.39 ep=131 | buf=35068
  Samples: old=31618, new=20000 | pseudo_fill=3250[W01] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=35068
  Samples: old=0, new=0 | new补充=200 | pseudo_fill=34868[W02] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=20200
  Samples: old=19882, new=20000[W03] stable(p=0.8192) | lwf=0.50 ep=90 | buf=20200
  Samples: old=20106, new=20000 | WARN: not enough non-rep old (94) | pseudo_fill=14868[W04] DRIFT(sev=0.25) | lwf=0.38 ep=134 | buf=35068
  Samples: old=0, new=0 | new补充=200 | pseudo_fill=34868[W05] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=20200
  Samples: old=5424, new=20000[W06] stable(p=0.8373) | lwf=0.50 ep=90 | buf=20200
  Samples: old=5397, new=20000 | pseudo_fill=29471[W07] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=35068
  Samples: old=12423, new=20000 | pseudo_fill=22445[W08] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=35068
  Samples: old=45, new=20000 | pseudo_fill=34823[W09] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=35068
  Samples: old=0, new=0 | new补充=200[W10] stable(p=0.5078) | lwf=0.50 ep=90 | buf=35068
  Samples: old=16679, new=2605[W11] stable(p=1.0000) | lwf=0.50 ep=90 | buf=35068
[After CL]
  Acc=0.7869  Pre=0.7208  Rec=0.9954  F1=0.8362
  Confusion: TP=44223 FP=17126 FN=204 TN=19786
Random seed set to: 5015

============================================================
Seed 5015 (5/5) | dataset=unsw | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8179  Pre=0.7551  Rec=0.9906  F1=0.8569
  Confusion: TP=44905 FP=14566 FN=427 TN=22434
  Samples: old=19856, new=20000[W00] stable(p=0.8992) | lwf=0.50 ep=90 | buf=35068
  Samples: old=0, new=20000[W01] stable(p=0.2990) | lwf=0.50 ep=90 | buf=35068
  Samples: old=0, new=20000 | pseudo_fill=34868[W02] DRIFT(sev=0.16) | lwf=0.42 ep=119 | buf=35068
  Samples: old=30592, new=20000[W03] stable(p=0.0648) | lwf=0.50 ep=90 | buf=35068
  Samples: old=9803, new=20000[W04] stable(p=0.5741) | lwf=0.50 ep=90 | buf=35068
  Samples: old=13926, new=20000[W05] stable(p=0.4758) | lwf=0.50 ep=90 | buf=35068
  Samples: old=311, new=20000[W06] stable(p=0.6987) | lwf=0.50 ep=90 | buf=35068
  Samples: old=226, new=20000 | pseudo_fill=34642[W07] DRIFT(sev=1.00) | lwf=0.00 ep=270 | buf=35068
  Samples: old=212, new=20000[W08] stable(p=0.4535) | lwf=0.50 ep=90 | buf=35068
  Samples: old=307, new=20000[W09] stable(p=0.3042) | lwf=0.50 ep=90 | buf=35068
  Samples: old=176, new=20000[W10] stable(p=0.3114) | lwf=0.50 ep=90 | buf=35068
  Samples: old=138, new=2605[W11] stable(p=1.0000) | lwf=0.50 ep=90 | buf=35068
[After CL]
  Acc=0.8191  Pre=0.7564  Rec=0.9856  F1=0.8559
  Confusion: TP=43703 FP=14078 FN=640 TN=22918