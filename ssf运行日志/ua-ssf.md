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
  Samples: old=24923, new=7 | new补充=43[W01] stable(p=0.4968) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25036, new=1 | new补充=49[W02] stable(p=0.5350) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24904, new=2 | new补充=48[W03] stable(p=0.3611) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24900, new=2 | new补充=48[W04] stable(p=0.2298) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25103, new=0 | new补充=50[W05] stable(p=0.1128) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24966, new=1 | new补充=49[W06] stable(p=0.4814) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24895, new=1 | new补充=49[W07] stable(p=0.1644) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25088, new=1 | new补充=49[W08] stable(p=0.8894) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25125, new=0 | new补充=50[W09] stable(p=0.9777) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25079, new=1 | new补充=49[W10] stable(p=0.8774) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24840, new=2 | new补充=48[W11] stable(p=0.0882) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25028, new=3 | new补充=47[W12] stable(p=0.9273) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24866, new=1 | new补充=49[W13] stable(p=0.2790) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24967, new=3 | new补充=47[W14] stable(p=0.4125) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25013, new=1 | new补充=49[W15] stable(p=0.5427) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25005, new=2 | new补充=48[W16] stable(p=0.2703) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24891, new=1 | new补充=49[W17] stable(p=0.1019) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25060, new=2 | new补充=48[W18] stable(p=0.9120) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25072, new=1 | new补充=49[W19] stable(p=0.9766) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24891, new=7 | new补充=43 | pseudo_fill=253[W20] DRIFT(sev=0.14) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24208, new=8 | new补充=42 | pseudo_fill=936[W21] DRIFT(sev=0.12) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24417, new=10 | new补充=40 | pseudo_fill=727[W22] DRIFT(sev=0.13) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24852, new=0 | new补充=50 | pseudo_fill=292[W23] DRIFT(sev=0.07) | lwf=0.00 ep=20 | buf=25194
  Samples: old=24980, new=2 | new补充=48[W24] stable(p=1.0000) | lwf=0.50 ep=20 | buf=25194
[After CL]
  Acc=0.8960  Pre=0.8968  Rec=0.9234  F1=0.9099
  Confusion: TP=11719 FP=1348 FN=972 TN=8267
Random seed set to: 5012

============================================================
Seed 5012 (2/5) | dataset=nsl | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8256  Pre=0.9108  Rec=0.7690  F1=0.8339
  Confusion: TP=9869 FP=967 FN=2964 TN=8744
  Samples: old=24945, new=5 | new补充=45[W00] stable(p=0.3017) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25134, new=1 | new补充=49[W01] stable(p=0.6563) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25067, new=1 | new补充=49[W02] stable(p=0.8929) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25069, new=1 | new补充=49[W03] stable(p=0.7686) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24859, new=5 | new补充=45[W04] stable(p=0.0658) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25031, new=1 | new补充=49[W05] stable(p=0.8170) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25026, new=0 | new补充=50[W06] stable(p=0.9005) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25088, new=2 | new补充=48[W07] stable(p=1.0000) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25106, new=3 | new补充=47[W08] stable(p=0.7524) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24842, new=4 | new补充=46[W09] stable(p=0.0964) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25034, new=3 | new补充=47[W10] stable(p=0.6411) | lwf=0.50 ep=20 | buf=25194
  [MINOR:sev=0.05]  Samples: old=24872, new=4 | new补充=46[W11] stable(p=0.0285) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25006, new=4 | new补充=46[W12] stable(p=0.7001) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25084, new=1 | new补充=49[W13] stable(p=0.9471) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25150, new=1 | WARN: not enough non-rep old to remove (44) | new补充=49[W14] stable(p=0.9507) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25107, new=1 | new补充=49[W15] stable(p=0.7134) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25085, new=1 | new补充=49[W16] stable(p=0.5852) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25097, new=1 | new补充=49[W17] stable(p=0.9687) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25089, new=3 | new补充=47[W18] stable(p=0.8731) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24969, new=2 | new补充=48[W19] stable(p=0.6898) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24849, new=2 | new补充=48 | pseudo_fill=295[W20] DRIFT(sev=0.11) | lwf=0.00 ep=21 | buf=25194
  Samples: old=23720, new=51 | pseudo_fill=1424[W21] DRIFT(sev=0.15) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24359, new=5 | new补充=45 | pseudo_fill=785[W22] DRIFT(sev=0.10) | lwf=0.00 ep=20 | buf=25194
  Samples: old=24670, new=7 | new补充=43 | pseudo_fill=474[W23] DRIFT(sev=0.08) | lwf=0.00 ep=20 | buf=25194
  Samples: old=24596, new=3 | new补充=47[W24] stable(p=1.0000) | lwf=0.50 ep=20 | buf=25194
[After CL]
  Acc=0.8975  Pre=0.8723  Rec=0.9604  F1=0.9142
  Confusion: TP=12189 FP=1784 FN=503 TN=7829
Random seed set to: 5013

============================================================
Seed 5013 (3/5) | dataset=nsl | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8637  Pre=0.8922  Rec=0.8651  F1=0.8785
  Confusion: TP=11102 FP=1341 FN=1731 TN=8370
  Samples: old=24963, new=5 | new补充=45[W00] stable(p=0.1832) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24939, new=1 | new补充=49[W01] stable(p=0.4653) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25140, new=0 | new补充=50[W02] stable(p=0.9816) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24990, new=3 | new补充=47[W03] stable(p=0.8288) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25022, new=1 | new补充=49[W04] stable(p=0.6599) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24997, new=2 | new补充=48[W05] stable(p=0.7149) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24960, new=2 | new补充=48[W06] stable(p=0.7105) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25125, new=2 | new补充=48[W07] stable(p=0.9739) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24954, new=4 | new补充=46[W08] stable(p=0.3671) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25029, new=1 | new补充=49[W09] stable(p=0.3543) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25088, new=0 | new补充=50[W10] stable(p=0.9680) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24900, new=8 | new补充=42[W11] stable(p=0.0867) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25117, new=2 | new补充=48[W12] stable(p=0.7719) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25126, new=3 | new补充=47[W13] stable(p=0.9954) | lwf=0.50 ep=20 | buf=25194
  [MINOR:sev=0.04]  Samples: old=24853, new=5 | new补充=45[W14] stable(p=0.0329) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25110, new=2 | new补充=48[W15] stable(p=0.8069) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24897, new=3 | new补充=47[W16] stable(p=0.1403) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25098, new=2 | new补充=48[W17] stable(p=0.9976) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24985, new=3 | new补充=47[W18] stable(p=0.7300) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24824, new=3 | new补充=47[W19] stable(p=0.1567) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24443, new=30 | new补充=20 | pseudo_fill=701[W20] DRIFT(sev=0.15) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24231, new=4 | new补充=46 | pseudo_fill=913[W21] DRIFT(sev=0.16) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24509, new=8 | new补充=42 | pseudo_fill=635[W22] DRIFT(sev=0.10) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24241, new=17 | new补充=33 | pseudo_fill=903[W23] DRIFT(sev=0.10) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24967, new=0 | new补充=50[W24] stable(p=1.0000) | lwf=0.50 ep=20 | buf=25194
[After CL]
  Acc=0.8967  Pre=0.8882  Rec=0.9364  F1=0.9117
  Confusion: TP=11904 FP=1498 FN=808 TN=8114
Random seed set to: 5014

============================================================
Seed 5014 (4/5) | dataset=nsl | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8529  Pre=0.9009  Rec=0.8333  F1=0.8658
  Confusion: TP=10694 FP=1177 FN=2139 TN=8534
  Samples: old=25129, new=1 | new补充=49[W00] stable(p=1.0000) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25101, new=2 | new补充=48[W01] stable(p=0.1299) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24974, new=1 | new补充=49[W02] stable(p=0.1750) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25150, new=1 | WARN: not enough non-rep old to remove (44) | new补充=49[W03] stable(p=0.9925) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25062, new=3 | new补充=47[W04] stable(p=0.5659) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25133, new=0 | new补充=50[W05] stable(p=0.9997) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25005, new=1 | new补充=49[W06] stable(p=0.4256) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25069, new=1 | new补充=49[W07] stable(p=0.9396) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25078, new=4 | new补充=46[W08] stable(p=0.2774) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25079, new=4 | new补充=46[W09] stable(p=0.8514) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25080, new=3 | new补充=47[W10] stable(p=0.9773) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25093, new=0 | new补充=50[W11] stable(p=0.3276) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24959, new=2 | new补充=48[W12] stable(p=0.4756) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25054, new=0 | new补充=50[W13] stable(p=0.6576) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25069, new=1 | new补充=49[W14] stable(p=0.9297) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25075, new=2 | new补充=48[W15] stable(p=0.6702) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24977, new=6 | new补充=44[W16] stable(p=0.3342) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25090, new=1 | new补充=49[W17] stable(p=0.6065) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25068, new=1 | new补充=49[W18] stable(p=0.6669) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25100, new=3 | new补充=47[W19] stable(p=0.8075) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25018, new=2 | new补充=48 | pseudo_fill=126[W20] DRIFT(sev=0.10) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24334, new=28 | new补充=22 | pseudo_fill=810[W21] DRIFT(sev=0.12) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24322, new=8 | new补充=42 | pseudo_fill=822[W22] DRIFT(sev=0.14) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24336, new=11 | new补充=39 | pseudo_fill=808[W23] DRIFT(sev=0.11) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24665, new=4 | new补充=46[W24] stable(p=1.0000) | lwf=0.50 ep=20 | buf=25194
[After CL]
  Acc=0.8920  Pre=0.9044  Rec=0.9063  F1=0.9054
  Confusion: TP=11519 FP=1217 FN=1191 TN=8378
Random seed set to: 5015

============================================================
Seed 5015 (5/5) | dataset=nsl | mode=ua-ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8250  Pre=0.8908  Rec=0.7894  F1=0.8370
  Confusion: TP=10130 FP=1242 FN=2703 TN=8469
  Samples: old=25008, new=1 | new补充=49[W00] stable(p=0.4142) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24916, new=2 | new补充=48[W01] stable(p=0.0966) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25127, new=0 | new补充=50[W02] stable(p=0.9916) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25025, new=2 | new补充=48[W03] stable(p=0.3170) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25086, new=0 | new补充=50[W04] stable(p=0.6521) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24878, new=1 | new补充=49[W05] stable(p=0.0751) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24939, new=2 | new补充=48[W06] stable(p=0.4049) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24850, new=3 | new补充=47[W07] stable(p=0.1401) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24911, new=4 | new补充=46[W08] stable(p=0.1587) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25096, new=1 | new补充=49[W09] stable(p=0.7943) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24998, new=2 | new补充=48[W10] stable(p=0.2901) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24650, new=3 | new补充=47 | pseudo_fill=494[W11] DRIFT(sev=0.06) | lwf=0.00 ep=20 | buf=25194
  Samples: old=25100, new=1 | new补充=49[W12] stable(p=0.9720) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24827, new=5 | new补充=45[W13] stable(p=0.0651) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25073, new=0 | new补充=50[W14] stable(p=0.7363) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24981, new=2 | new补充=48[W15] stable(p=0.3773) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25034, new=2 | new补充=48[W16] stable(p=0.2889) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25051, new=3 | new补充=47[W17] stable(p=0.9405) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25011, new=1 | new补充=49[W18] stable(p=0.2305) | lwf=0.50 ep=20 | buf=25194
  Samples: old=25103, new=1 | new补充=49[W19] stable(p=0.0808) | lwf=0.50 ep=20 | buf=25194
  Samples: old=24672, new=17 | new补充=33 | pseudo_fill=472[W20] DRIFT(sev=0.09) | lwf=0.00 ep=20 | buf=25194
  Samples: old=24430, new=3 | new补充=47 | pseudo_fill=714[W21] DRIFT(sev=0.12) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24637, new=21 | new补充=29 | pseudo_fill=507[W22] DRIFT(sev=0.05) | lwf=0.00 ep=20 | buf=25194
  Samples: old=24584, new=9 | new补充=41 | pseudo_fill=560[W23] DRIFT(sev=0.13) | lwf=0.00 ep=21 | buf=25194
  Samples: old=24736, new=3 | new补充=47[W24] stable(p=1.0000) | lwf=0.50 ep=20 | buf=25194
[After CL]
  Acc=0.8917  Pre=0.8708  Rec=0.9507  F1=0.9090
  Confusion: TP=12069 FP=1790 FN=626 TN=7832


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
