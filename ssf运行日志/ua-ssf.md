```bash
python ssf.py --mode ua-ssf --dataset nsl --epochs 200 --epoch_1 20 --sample_interval 5000 --num_labeled_sample 50 --opt_old_lr 100 --opt_new_lr 8 --new_sample_weight 3 --cuda 0
```

Random seed set to: 5011

============================================================
Seed 5011 (1/5) | dataset=nsl | mode=ua-ssf
============================================================
[Config] train=125973, test=22544, memory=25194, input_dim=121
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8793  Pre=0.8862  Rec=0.9040  F1=0.8950
  Confusion: TP=11601 FP=1490 FN=1232 TN=8221
  Samples: old=25059, new=0 | new补充=50[W00] stable(p=0.9730) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24948, new=3 | new补充=47[W01] stable(p=0.5126) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25060, new=1 | new补充=49[W02] stable(p=0.7632) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24893, new=2 | new补充=48[W03] stable(p=0.3299) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24892, new=2 | new补充=48[W04] stable(p=0.2863) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25101, new=1 | new补充=49[W05] stable(p=0.1026) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24962, new=0 | new补充=50[W06] stable(p=0.5891) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24882, new=5 | new补充=45[W07] stable(p=0.1882) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25105, new=2 | new补充=48[W08] stable(p=0.8242) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25138, new=1 | new补充=49[W09] stable(p=0.8025) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25148, new=0 | WARN: not enough non-rep old to remove (46) | new补充=50[W10] stable(p=0.9053) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24841, new=4 | new补充=46[W11] stable(p=0.1251) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25054, new=1 | new补充=49[W12] stable(p=0.7245) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24869, new=1 | new补充=49[W13] stable(p=0.2218) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24915, new=3 | new补充=47[W14] stable(p=0.2676) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25086, new=3 | new补充=47[W15] stable(p=0.8257) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25079, new=0 | new补充=50[W16] stable(p=0.2313) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24869, new=4 | new补充=46[W17] stable(p=0.1473) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24971, new=2 | new补充=48[W18] stable(p=0.8995) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25025, new=4 | new补充=46[W19] stable(p=0.8843) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24949, new=4 | new补充=46 | pseudo_fill=195[W20] DRIFT(sev=0.12) | lwf=0.00 ep=22 | buf=25194
  Samples: old=24655, new=10 | new补充=40 | pseudo_fill=489[W21] DRIFT(sev=0.11) | lwf=0.00 ep=22 | buf=25194
  Samples: old=23896, new=7 | new补充=43 | pseudo_fill=1248[W22] DRIFT(sev=0.13) | lwf=0.00 ep=22 | buf=25194
  Samples: old=24785, new=2 | new补充=48 | pseudo_fill=359[W23] DRIFT(sev=0.09) | lwf=0.00 ep=21 | buf=25194
  Samples: old=25070, new=1 | new补充=49[W24] stable(p=1.0000) | lwf=0.50 ep=20 | buf=25194
[After CL]
  Acc=0.8944  Pre=0.8757  Rec=0.9493  F1=0.9110
  Confusion: TP=12054 FP=1711 FN=644 TN=7895
Random seed set to: 5012

============================================================
Seed 5012 (2/5) | dataset=nsl | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8256  Pre=0.9108  Rec=0.7690  F1=0.8339
  Confusion: TP=9869 FP=967 FN=2964 TN=8744
  Samples: old=24945, new=5 | new补充=45[W00] stable(p=0.3017) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25140, new=0 | new补充=50[W01] stable(p=0.5923) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25049, new=4 | new补充=46[W02] stable(p=0.9293) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25112, new=0 | new补充=50[W03] stable(p=0.8557) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24779, new=3 | new补充=47[W04] stable(p=0.1046) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24999, new=2 | new补充=48[W05] stable(p=0.9143) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24994, new=1 | new补充=49[W06] stable(p=0.8950) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25044, new=1 | new补充=49[W07] stable(p=0.9943) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25075, new=3 | new补充=47[W08] stable(p=0.9955) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24814, new=5 | new补充=45[W09] stable(p=0.0765) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25139, new=0 | new补充=50[W10] stable(p=0.8126) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24878, new=5 | new补充=45[W11] stable(p=0.0651) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24975, new=0 | new补充=50[W12] stable(p=0.7504) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25051, new=1 | new补充=49[W13] stable(p=0.9946) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25131, new=0 | new补充=50[W14] stable(p=0.9833) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25057, new=1 | new补充=49[W15] stable(p=0.9338) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25070, new=1 | new补充=49[W16] stable(p=0.8641) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25101, new=2 | new补充=48[W17] stable(p=0.9907) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25034, new=1 | new补充=49[W18] stable(p=0.9792) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24979, new=3 | new补充=47[W19] stable(p=0.5638) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24706, new=5 | new补充=45 | pseudo_fill=438[W20] DRIFT(sev=0.08) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24826, new=6 | new补充=44 | pseudo_fill=318[W21] DRIFT(sev=0.13) | lwf=0.00 ep=22 | buf=25194
  Samples: old=24802, new=3 | new补充=47 | pseudo_fill=342[W22] DRIFT(sev=0.10) | lwf=0.00 ep=22 | buf=25194
  Samples: old=24109, new=29 | new补充=21 | pseudo_fill=1035[W23] DRIFT(sev=0.10) | lwf=0.00 ep=22 | buf=25194
  Samples: old=24329, new=5 | new补充=45[W24] stable(p=1.0000) | lwf=0.50 ep=20 | buf=25194
[After CL]
  Acc=0.8735  Pre=0.8807  Rec=0.8998  F1=0.8901
  Confusion: TP=11432 FP=1549 FN=1273 TN=8054
Random seed set to: 5013

============================================================
Seed 5013 (3/5) | dataset=nsl | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8637  Pre=0.8922  Rec=0.8651  F1=0.8785
  Confusion: TP=11102 FP=1341 FN=1731 TN=8370
  Samples: old=24963, new=5 | new补充=45[W00] stable(p=0.1832) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24944, new=3 | new补充=47[W01] stable(p=0.4674) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25158, new=1 | WARN: not enough non-rep old to remove (36) | new补充=49[W02] stable(p=0.9970) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25046, new=2 | new补充=48[W03] stable(p=0.8756) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24972, new=0 | new补充=50[W04] stable(p=0.5712) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25029, new=4 | new补充=46[W05] stable(p=0.5791) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25027, new=0 | new补充=50[W06] stable(p=0.8459) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25116, new=1 | new补充=49[W07] stable(p=0.9054) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24972, new=0 | new补充=50[W08] stable(p=0.4145) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25017, new=1 | new补充=49[W09] stable(p=0.3193) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25112, new=0 | new补充=50[W10] stable(p=0.9318) | lwf=0.50 ep=20 | buf=25194
  [MINOR:sev=0.05]  Samples: old=24961, new=1 | new补充=49[W11] stable(p=0.0160) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25136, new=0 | new补充=50[W12] stable(p=0.8260) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25113, new=0 | new补充=50[W13] stable(p=0.8285) | lwf=0.50 ep=20 | buf=25194
  [MINOR:sev=0.04]  Samples: old=24870, new=1 | new补充=49[W14] stable(p=0.0306) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25154, new=1 | WARN: not enough non-rep old to remove (40) | new补充=49[W15] stable(p=0.6938) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24903, new=3 | new补充=47[W16] stable(p=0.0964) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25121, new=2 | new补充=48[W17] stable(p=0.9919) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25047, new=0 | new补充=50[W18] stable(p=0.7433) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24813, new=4 | new补充=46[W19] stable(p=0.1097) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24794, new=4 | new补充=46 | pseudo_fill=350[W20] DRIFT(sev=0.13) | lwf=0.00 ep=22 | buf=25194
  Samples: old=24343, new=51 | pseudo_fill=801[W21] DRIFT(sev=0.13) | lwf=0.00 ep=22 | buf=25194
  Samples: old=24537, new=15 | new补充=35 | pseudo_fill=607[W22] DRIFT(sev=0.10) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24399, new=10 | new补充=40 | pseudo_fill=745[W23] DRIFT(sev=0.08) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24922, new=2 | new补充=48[W24] stable(p=1.0000) | lwf=0.50 ep=20 | buf=25194
[After CL]
  Acc=0.8918  Pre=0.8818  Rec=0.9353  F1=0.9077
  Confusion: TP=11876 FP=1592 FN=822 TN=8018
Random seed set to: 5014

============================================================
Seed 5014 (4/5) | dataset=nsl | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8529  Pre=0.9009  Rec=0.8333  F1=0.8658
  Confusion: TP=10694 FP=1177 FN=2139 TN=8534
  Samples: old=25129, new=1 | new补充=49[W00] stable(p=1.0000) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25085, new=2 | new补充=48[W01] stable(p=0.1806) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24952, new=1 | new补充=49[W02] stable(p=0.2780) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25147, new=1 | WARN: not enough non-rep old to remove (47) | new补充=49[W03] stable(p=0.9701) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25082, new=1 | new补充=49[W04] stable(p=0.5322) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25103, new=0 | new补充=50[W05] stable(p=0.9911) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24975, new=4 | new补充=46[W06] stable(p=0.6018) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25094, new=2 | new补充=48[W07] stable(p=0.9947) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25071, new=0 | new补充=50[W08] stable(p=0.4334) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25082, new=1 | new补充=49[W09] stable(p=0.6261) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25084, new=1 | new补充=49[W10] stable(p=0.9639) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25081, new=0 | new补充=50[W11] stable(p=0.3214) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24957, new=4 | new补充=46[W12] stable(p=0.4135) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25078, new=1 | new补充=49[W13] stable(p=0.7140) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25110, new=1 | new补充=49[W14] stable(p=0.8049) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25033, new=3 | new补充=47[W15] stable(p=0.4571) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25005, new=0 | new补充=50[W16] stable(p=0.4875) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25041, new=2 | new补充=48[W17] stable(p=0.5752) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24985, new=7 | new补充=43[W18] stable(p=0.6873) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25006, new=0 | new补充=50[W19] stable(p=0.8227) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24812, new=5 | new补充=45 | pseudo_fill=332[W20] DRIFT(sev=0.07) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24474, new=7 | new补充=43 | pseudo_fill=670[W21] DRIFT(sev=0.08) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24675, new=7 | new补充=43 | pseudo_fill=469[W22] DRIFT(sev=0.08) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24055, new=23 | new补充=27 | pseudo_fill=1089[W23] DRIFT(sev=0.11) | lwf=0.00 ep=22 | buf=25194
  Samples: old=24574, new=6 | new补充=44[W24] stable(p=1.0000) | lwf=0.50 ep=20 | buf=25194
[After CL]
  Acc=0.8986  Pre=0.8838  Rec=0.9464  F1=0.9140
  Confusion: TP=12030 FP=1582 FN=681 TN=8016
Random seed set to: 5015

============================================================
Seed 5015 (5/5) | dataset=nsl | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8250  Pre=0.8908  Rec=0.7894  F1=0.8370
  Confusion: TP=10130 FP=1242 FN=2703 TN=8469
  Samples: old=25008, new=1 | new补充=49[W00] stable(p=0.4142) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24882, new=0 | new补充=50[W01] stable(p=0.0872) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25130, new=1 | new补充=49[W02] stable(p=0.9810) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24985, new=1 | new补充=49[W03] stable(p=0.3330) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25100, new=2 | new补充=48[W04] stable(p=0.5186) | lwf=0.50 ep=20 | buf=25194
  [MINOR:sev=0.04]  Samples: old=24903, new=2 | new补充=48[W05] stable(p=0.0322) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24968, new=3 | new补充=47[W06] stable(p=0.4194) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24883, new=1 | new补充=49[W07] stable(p=0.1567) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24897, new=0 | new补充=50[W08] stable(p=0.3101) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25116, new=0 | new补充=50[W09] stable(p=0.8292) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25001, new=5 | new补充=45[W10] stable(p=0.3519) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24678, new=4 | new补充=46 | pseudo_fill=466[W11] DRIFT(sev=0.06) | lwf=0.00 ep=21 | buf=25194
  Samples: old=25086, new=1 | new补充=49[W12] stable(p=0.9649) | lwf=0.50 ep=20 | buf=25194
  [MINOR:sev=0.05]  Samples: old=24675, new=6 | new补充=44[W13] stable(p=0.0155) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25008, new=2 | new补充=48[W14] stable(p=0.7611) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24959, new=1 | new补充=49[W15] stable(p=0.0625) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25035, new=1 | new补充=49[W16] stable(p=0.7363) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25099, new=0 | new补充=50[W17] stable(p=0.9264) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24880, new=5 | new补充=45[W18] stable(p=0.2966) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24939, new=1 | new补充=49[W19] stable(p=0.1074) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24900, new=10 | new补充=40 | pseudo_fill=244[W20] DRIFT(sev=0.06) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24960, new=1 | new补充=49 | pseudo_fill=184[W21] DRIFT(sev=0.06) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24519, new=26 | new补充=24 | pseudo_fill=625[W22] DRIFT(sev=0.07) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24524, new=11 | new补充=39 | pseudo_fill=620[W23] DRIFT(sev=0.11) | lwf=0.00 ep=22 | buf=25194
  Samples: old=24354, new=6 | new补充=44[W24] stable(p=1.0000) | lwf=0.50 ep=20 | buf=25194
[After CL]
  Acc=0.8871  Pre=0.8714  Rec=0.9405  F1=0.9046
  Confusion: TP=11941 FP=1763 FN=756 TN=7854

```bash
python ssf.py --mode ua-ssf --dataset unsw --epochs 200 --epoch_1 180 --sample_interval 20000 --num_labeled_sample 200 --opt_old_lr 24 --opt_new_lr 50 --new_sample_weight 60 --uncertainty_beta 0 --cuda 0
```

Random seed set to: 5011

============================================================
Seed 5011 (1/5) | dataset=unsw | mode=ua-ssf
============================================================
[Config] train=175341, test=82332, memory=35068, input_dim=196
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8185  Pre=0.7557  Rec=0.9907  F1=0.8574
  Confusion: TP=44909 FP=14519 FN=423 TN=22481
  Samples: old=34996, new=48 | WARN: not enough non-rep old to remove (72) | new补充=152[W00] stable(p=0.1100) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35024, new=32 | WARN: not enough non-rep old to remove (44) | new补充=168[W01] stable(p=0.8322) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35018, new=32 | WARN: not enough non-rep old to remove (50) | new补充=168[W02] stable(p=0.2278) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35041, new=28 | WARN: not enough non-rep old to remove (27) | new补充=172[W03] stable(p=0.8362) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35048, new=14 | WARN: not enough non-rep old to remove (20) | new补充=186[W04] stable(p=0.8506) | lwf=0.50 ep=180 | buf=35068
  [MINOR:sev=0.03]  Samples: old=34975, new=65 | WARN: not enough non-rep old to remove (93) | new补充=135[W05] stable(p=0.0010) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35009, new=44 | WARN: not enough non-rep old to remove (59) | new补充=156[W06] stable(p=0.6442) | lwf=0.50 ep=180 | buf=35068
  Samples: old=34806, new=198 | new补充=2 | pseudo_fill=62[W07] DRIFT(sev=0.24) | lwf=0.00 ep=222 | buf=35068
  Samples: old=34701, new=364 | pseudo_fill=167[W08] DRIFT(sev=0.24) | lwf=0.00 ep=223 | buf=35068
  Samples: old=34803, new=211 | pseudo_fill=65[W09] DRIFT(sev=0.25) | lwf=0.00 ep=224 | buf=35068
  Samples: old=34885, new=134 | WARN: not enough non-rep old (183) | new补充=66[W10] DRIFT(sev=0.23) | lwf=0.00 ep=220 | buf=35068
  Samples: old=34879, new=39 | WARN: not enough non-rep old to remove (189) | new补充=161[W11] stable(p=1.0000) | lwf=0.50 ep=180 | buf=35068
[After CL]
  Acc=0.8993  Pre=0.8695  Rec=0.9614  F1=0.9131
  Confusion: TP=43168 FP=6480 FN=1734 TN=30151
Random seed set to: 5012

============================================================
Seed 5012 (2/5) | dataset=unsw | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8178  Pre=0.7554  Rec=0.9895  F1=0.8567
  Confusion: TP=44857 FP=14527 FN=475 TN=22473
  Samples: old=35037, new=27 | WARN: not enough non-rep old to remove (31) | new补充=173[W00] stable(p=0.6201) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35016, new=43 | WARN: not enough non-rep old to remove (52) | new补充=157[W01] stable(p=0.3731) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35038, new=27 | WARN: not enough non-rep old to remove (30) | new补充=173[W02] stable(p=0.6933) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35036, new=50 | WARN: not enough non-rep old to remove (32) | new补充=150[W03] stable(p=0.5277) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35025, new=23 | WARN: not enough non-rep old to remove (43) | new补充=177[W04] stable(p=0.6747) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35034, new=17 | WARN: not enough non-rep old to remove (34) | new补充=183[W05] stable(p=0.9615) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35030, new=33 | WARN: not enough non-rep old to remove (38) | new补充=167[W06] stable(p=0.3177) | lwf=0.50 ep=180 | buf=35068
  Samples: old=34643, new=415 | pseudo_fill=225[W07] DRIFT(sev=0.23) | lwf=0.00 ep=222 | buf=35068
  Samples: old=34700, new=361 | pseudo_fill=168[W08] DRIFT(sev=0.20) | lwf=0.00 ep=215 | buf=35068
  Samples: old=34863, new=198 | new补充=2 | pseudo_fill=5[W09] DRIFT(sev=0.17) | lwf=0.00 ep=210 | buf=35068
  Samples: old=34822, new=215 | pseudo_fill=46[W10] DRIFT(sev=0.16) | lwf=0.00 ep=209 | buf=35068
  Samples: old=34926, new=23 | WARN: not enough non-rep old to remove (142) | new补充=177[W11] stable(p=1.0000) | lwf=0.50 ep=180 | buf=35068
[After CL]
  Acc=0.8933  Pre=0.8562  Rec=0.9689  F1=0.9091
  Confusion: TP=43469 FP=7298 FN=1393 TN=29312
Random seed set to: 5013

============================================================
Seed 5013 (3/5) | dataset=unsw | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8175  Pre=0.7552  Rec=0.9894  F1=0.8565
  Confusion: TP=44850 FP=14542 FN=482 TN=22458
  Samples: old=35026, new=31 | WARN: not enough non-rep old to remove (42) | new补充=169[W00] stable(p=0.8332) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35027, new=39 | WARN: not enough non-rep old to remove (41) | new补充=161[W01] stable(p=0.4097) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35025, new=38 | WARN: not enough non-rep old to remove (43) | new补充=162[W02] stable(p=0.2564) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35021, new=24 | WARN: not enough non-rep old to remove (47) | new补充=176[W03] stable(p=0.8121) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35033, new=14 | WARN: not enough non-rep old to remove (35) | new补充=186[W04] stable(p=0.6552) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35024, new=35 | WARN: not enough non-rep old to remove (44) | new补充=165[W05] stable(p=0.4507) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35040, new=19 | WARN: not enough non-rep old to remove (28) | new补充=181[W06] stable(p=0.9933) | lwf=0.50 ep=180 | buf=35068
  Samples: old=34772, new=272 | pseudo_fill=96[W07] DRIFT(sev=0.23) | lwf=0.00 ep=222 | buf=35068
  Samples: old=34747, new=282 | pseudo_fill=121[W08] DRIFT(sev=0.20) | lwf=0.00 ep=216 | buf=35068
  Samples: old=34795, new=251 | pseudo_fill=73[W09] DRIFT(sev=0.18) | lwf=0.00 ep=213 | buf=35068
  Samples: old=34783, new=237 | pseudo_fill=85[W10] DRIFT(sev=0.19) | lwf=0.00 ep=214 | buf=35068
  Samples: old=34906, new=24 | WARN: not enough non-rep old to remove (162) | new补充=176[W11] stable(p=1.0000) | lwf=0.50 ep=180 | buf=35068
[After CL]
  Acc=0.9098  Pre=0.8811  Rec=0.9666  F1=0.9219
  Confusion: TP=43394 FP=5857 FN=1498 TN=30785
Random seed set to: 5014

============================================================
Seed 5014 (4/5) | dataset=unsw | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8162  Pre=0.7522  Rec=0.9935  F1=0.8562
  Confusion: TP=45038 FP=14835 FN=294 TN=22165
  [MINOR:sev=0.03]  Samples: old=35016, new=47 | WARN: not enough non-rep old to remove (52) | new补充=153[W00] stable(p=0.0052) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35009, new=47 | WARN: not enough non-rep old to remove (59) | new补充=153[W01] stable(p=0.1048) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35034, new=37 | WARN: not enough non-rep old to remove (34) | new补充=163[W02] stable(p=0.9863) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35050, new=23 | WARN: not enough non-rep old to remove (18) | new补充=177[W03] stable(p=0.9210) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35048, new=10 | WARN: not enough non-rep old to remove (20) | new补充=190[W04] stable(p=0.8577) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35036, new=24 | WARN: not enough non-rep old to remove (32) | new补充=176[W05] stable(p=0.9567) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35038, new=22 | WARN: not enough non-rep old to remove (30) | new补充=178[W06] stable(p=0.4955) | lwf=0.50 ep=180 | buf=35068
  Samples: old=34800, new=259 | pseudo_fill=68[W07] DRIFT(sev=0.23) | lwf=0.00 ep=221 | buf=35068
  Samples: old=34681, new=318 | pseudo_fill=187[W08] DRIFT(sev=0.24) | lwf=0.00 ep=222 | buf=35068
  Samples: old=34867, new=213 | pseudo_fill=1[W09] DRIFT(sev=0.21) | lwf=0.00 ep=218 | buf=35068
  Samples: old=34727, new=295 | pseudo_fill=141[W10] DRIFT(sev=0.19) | lwf=0.00 ep=213 | buf=35068
  Samples: old=34712, new=53 | new补充=147[W11] stable(p=1.0000) | lwf=0.50 ep=180 | buf=35068
[After CL]
  Acc=0.8895  Pre=0.8554  Rec=0.9619  F1=0.9055
  Confusion: TP=43181 FP=7298 FN=1710 TN=29346
Random seed set to: 5015

============================================================
Seed 5015 (5/5) | dataset=unsw | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8179  Pre=0.7551  Rec=0.9906  F1=0.8569
  Confusion: TP=44905 FP=14566 FN=427 TN=22434
  Samples: old=35013, new=36 | WARN: not enough non-rep old to remove (55) | new补充=164[W00] stable(p=0.8992) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35029, new=44 | WARN: not enough non-rep old to remove (39) | new补充=156[W01] stable(p=0.5421) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35032, new=27 | WARN: not enough non-rep old to remove (36) | new补充=173[W02] stable(p=0.8377) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35019, new=54 | WARN: not enough non-rep old to remove (49) | new补充=146[W03] stable(p=0.3819) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35008, new=49 | WARN: not enough non-rep old to remove (60) | new补充=151[W04] stable(p=0.2664) | lwf=0.50 ep=180 | buf=35068
  Samples: old=35030, new=35 | WARN: not enough non-rep old to remove (38) | new补充=165[W05] stable(p=0.6792) | lwf=0.50 ep=180 | buf=35068
  Samples: old=34996, new=60 | WARN: not enough non-rep old to remove (72) | new补充=140[W06] stable(p=0.2990) | lwf=0.50 ep=180 | buf=35068
  Samples: old=34834, new=205 | pseudo_fill=34[W07] DRIFT(sev=0.20) | lwf=0.00 ep=215 | buf=35068
  Samples: old=34604, new=413 | pseudo_fill=264[W08] DRIFT(sev=0.23) | lwf=0.00 ep=220 | buf=35068
  Samples: old=34894, new=120 | WARN: not enough non-rep old (174) | new补充=80[W09] DRIFT(sev=0.20) | lwf=0.00 ep=215 | buf=35068
  Samples: old=34873, new=212 | WARN: not enough non-rep old (195)[W10] DRIFT(sev=0.20) | lwf=0.00 ep=216 | buf=35068
  Samples: old=34687, new=51 | new补充=149[W11] stable(p=1.0000) | lwf=0.50 ep=180 | buf=35068
[After CL]
  Acc=0.8980  Pre=0.8810  Rec=0.9420  F1=0.9105
  Confusion: TP=42294 FP=5715 FN=2602 TN=30926
