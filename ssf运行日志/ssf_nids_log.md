###### 1.nsl运行

(ssf) jackson@omnisky:~/wkcodenew/SSF-Strategic-Selection-and-Forgetting-main$ python ssf.py --mode s
sf --dataset nsl --epochs 200 --epoch_1 20 --sample_interval 5000 --num_labeled_sample 50 --opt_old_l
r 100 --opt_new_lr 8 --new_sample_weight 3 --cuda 0
Random seed set to: 5011

============================================================
Seed 5011 (1/5) | dataset=nsl | mode=ssf
============================================================
[Config] train=125973, test=22544, memory=25194, input_dim=121
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8724  Pre=0.8846  Rec=0.8922  F1=0.8884
  Confusion: TP=11450 FP=1493 FN=1383 TN=8218
  Samples: old=25036, new=3 | new补充=47[W00] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25002, new=2 | new补充=48[W01] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25007, new=3 | new补充=47[W02] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24942, new=0 | new补充=50[W03] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24987, new=5 | new补充=45[W04] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25111, new=1 | new补充=49[W05] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24928, new=6 | new补充=44[W06] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24809, new=2 | new补充=48[W07] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25081, new=0 | new补充=50[W08] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25095, new=0 | new补充=50[W09] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25068, new=0 | new补充=50[W10] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24878, new=3 | new补充=47[W11] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24968, new=0 | new补充=50[W12] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24818, new=2 | new补充=48[W13] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24880, new=5 | new补充=45[W14] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25060, new=2 | new补充=48[W15] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25091, new=0 | new补充=50[W16] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24887, new=0 | new补充=50[W17] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25010, new=1 | new补充=49[W18] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25031, new=3 | new补充=47[W19] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24964, new=5 | new补充=45 | pseudo_fill=180[W20] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24576, new=10 | new补充=40 | pseudo_fill=568[W21] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24327, new=13 | new补充=37 | pseudo_fill=817[W22] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24254, new=28 | new补充=22 | pseudo_fill=890[W23] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24909, new=2 | new补充=48[W24] stable | lwf=0.50 ep=20 | buf=25194
[After CL]
  Acc=0.8981  Pre=0.8992  Rec=0.9248  F1=0.9118
  Confusion: TP=11754 FP=1318 FN=956 TN=8279
Random seed set to: 5012

============================================================
Seed 5012 (2/5) | dataset=nsl | mode=ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8270  Pre=0.9042  Rec=0.7785  F1=0.8367
  Confusion: TP=9991 FP=1058 FN=2842 TN=8653
  Samples: old=24898, new=6 | new补充=44[W00] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25071, new=3 | new补充=47[W01] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25041, new=4 | new补充=46[W02] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24997, new=1 | new补充=49[W03] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24848, new=2 | new补充=48[W04] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25003, new=2 | new补充=48[W05] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25036, new=0 | new补充=50[W06] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25099, new=0 | new补充=50[W07] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25075, new=0 | new补充=50[W08] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24808, new=3 | new补充=47[W09] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25010, new=2 | new补充=48[W10] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24847, new=2 | new补充=48[W11] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24998, new=2 | new补充=48[W12] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25088, new=0 | new补充=50[W13] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25044, new=5 | new补充=45[W14] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25129, new=0 | new补充=50[W15] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25043, new=2 | new补充=48[W16] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25057, new=2 | new补充=48[W17] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25137, new=1 | new补充=49[W18] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24924, new=5 | new补充=45[W19] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24928, new=3 | new补充=47 | pseudo_fill=216[W20] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24144, new=14 | new补充=36 | pseudo_fill=1000[W21] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24552, new=8 | new补充=42 | pseudo_fill=592[W22] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24683, new=3 | new补充=47 | pseudo_fill=461[W23] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24212, new=21 | new补充=29[W24] stable | lwf=0.50 ep=20 | buf=25194
[After CL]
  Acc=0.8964  Pre=0.8919  Rec=0.9307  F1=0.9109
  Confusion: TP=11812 FP=1432 FN=880 TN=8184
Random seed set to: 5013

============================================================
Seed 5013 (3/5) | dataset=nsl | mode=ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8658  Pre=0.8997  Rec=0.8601  F1=0.8795
  Confusion: TP=11038 FP=1231 FN=1795 TN=8480
  Samples: old=24984, new=0 | new补充=50[W00] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24932, new=2 | new补充=48[W01] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25134, new=0 | new补充=50[W02] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25037, new=2 | new补充=48[W03] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25069, new=2 | new补充=48[W04] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24987, new=1 | new补充=49[W05] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24990, new=1 | new补充=49[W06] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25138, new=1 | new补充=49[W07] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24963, new=4 | new补充=46[W08] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25035, new=4 | new补充=46[W09] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25118, new=0 | new补充=50[W10] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24981, new=5 | new补充=45[W11] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25054, new=1 | new补充=49[W12] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25129, new=0 | new补充=50[W13] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24858, new=6 | new补充=44 | pseudo_fill=286[W14] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=25131, new=1 | new补充=49[W15] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25018, new=3 | new补充=47[W16] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25110, new=0 | new补充=50[W17] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25149, new=0 | WARN: not enough non-rep old to remove (45) | new补充=50[W18] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24944, new=2 | new补充=48[W19] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24067, new=12 | new补充=38 | pseudo_fill=1077[W20] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24317, new=7 | new补充=43 | pseudo_fill=827[W21] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24868, new=2 | new补充=48 | pseudo_fill=276[W22] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24446, new=10 | new补充=40 | pseudo_fill=698[W23] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24917, new=2 | new补充=48[W24] stable | lwf=0.50 ep=20 | buf=25194
[After CL]
  Acc=0.9003  Pre=0.8761  Rec=0.9604  F1=0.9164
  Confusion: TP=12188 FP=1723 FN=502 TN=7902
Random seed set to: 5014

============================================================
Seed 5014 (4/5) | dataset=nsl | mode=ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8417  Pre=0.9004  Rec=0.8117  F1=0.8537
  Confusion: TP=10416 FP=1152 FN=2417 TN=8559
  Samples: old=25116, new=2 | new补充=48[W00] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25046, new=1 | new补充=49[W01] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24931, new=4 | new补充=46[W02] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25074, new=0 | new补充=50[W03] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25027, new=1 | new补充=49[W04] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25124, new=0 | new补充=50[W05] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25085, new=0 | new补充=50[W06] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25097, new=1 | new补充=49[W07] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25076, new=3 | new补充=47[W08] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25142, new=1 | new补充=49[W09] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25078, new=3 | new补充=47[W10] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25006, new=5 | new补充=45[W11] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25011, new=3 | new补充=47[W12] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25096, new=1 | new补充=49[W13] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25052, new=2 | new补充=48[W14] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25052, new=2 | new补充=48[W15] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25060, new=2 | new补充=48[W16] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24981, new=6 | new补充=44[W17] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25085, new=2 | new补充=48[W18] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25121, new=1 | new补充=49[W19] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24655, new=10 | new补充=40 | pseudo_fill=489[W20] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24595, new=6 | new补充=44 | pseudo_fill=549[W21] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24349, new=10 | new补充=40 | pseudo_fill=795[W22] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24381, new=10 | new补充=40 | pseudo_fill=763[W23] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24840, new=1 | new补充=49[W24] stable | lwf=0.50 ep=20 | buf=25194
[After CL]
  Acc=0.9036  Pre=0.8881  Rec=0.9504  F1=0.9182
  Confusion: TP=12069 FP=1520 FN=630 TN=8094
Random seed set to: 5015

============================================================
Seed 5015 (5/5) | dataset=nsl | mode=ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8424  Pre=0.8738  Rec=0.8452  F1=0.8593
  Confusion: TP=10846 FP=1566 FN=1987 TN=8145
  Samples: old=25032, new=2 | new补充=48[W00] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24923, new=1 | new补充=49[W01] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25151, new=2 | WARN: not enough non-rep old to remove (43) | new补充=48[W02] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24970, new=3 | new补充=47[W03] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25132, new=2 | new补充=48[W04] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24795, new=3 | new补充=47[W05] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24936, new=3 | new补充=47[W06] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24927, new=3 | new补充=47[W07] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24958, new=3 | new补充=47[W08] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25058, new=1 | new补充=49[W09] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25034, new=0 | new补充=50[W10] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24715, new=4 | new补充=46 | pseudo_fill=429[W11] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=25117, new=1 | new补充=49[W12] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24918, new=1 | new补充=49[W13] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25047, new=1 | new补充=49[W14] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25098, new=2 | new补充=48[W15] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25088, new=2 | new补充=48[W16] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=25144, new=0 | new补充=50[W17] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24859, new=8 | new补充=42 | pseudo_fill=285[W18] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=25096, new=3 | new补充=47[W19] stable | lwf=0.50 ep=20 | buf=25194
  Samples: old=24211, new=34 | new补充=16 | pseudo_fill=933[W20] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=23868, new=11 | new补充=39 | pseudo_fill=1276[W21] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=25050, new=0 | new补充=50 | pseudo_fill=94[W22] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=24290, new=22 | new补充=28 | pseudo_fill=854[W23] DRIFT | lwf=0.00 ep=20 | buf=25194
  Samples: old=25048, new=0 | new补充=50[W24] stable | lwf=0.50 ep=20 | buf=25194
[After CL]
  Acc=0.9046  Pre=0.8791  Rec=0.9651  F1=0.9201
  Confusion: TP=12265 FP=1687 FN=444 TN=7934




###### 2.usw
(ssf) jackson@omnisky:~/wkcodenew/SSF-Strategic-Selection-and-Forgetting-main$ python ssf.py --mode ssf --dataset unsw --epochs 200 --epoch_1 180 --sample_interval 20000 --num_labeled_sample 200 --opt_old_lr 24 --opt_new_lr 50 --new_sample_weight 60 --cuda 0
Random seed set to: 5011

============================================================
Seed 5011 (1/5) | dataset=unsw | mode=ssf
============================================================
[Config] train=175341, test=82332, memory=35068, input_dim=196
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8191  Pre=0.7565  Rec=0.9901  F1=0.8577
  Confusion: TP=44885 FP=14445 FN=447 TN=22555
  Samples: old=35012, new=59 | WARN: not enough non-rep old to remove (56) | new补充=141[W00] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35031, new=35 | WARN: not enough non-rep old to remove (37) | new补充=165[W01] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35001, new=58 | WARN: not enough non-rep old to remove (67) | new补充=142[W02] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35040, new=25 | WARN: not enough non-rep old to remove (28) | new补充=175[W03] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35038, new=41 | WARN: not enough non-rep old to remove (30) | new补充=159[W04] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=34947, new=94 | WARN: not enough non-rep old (121) | new补充=106[W05] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=35036, new=26 | WARN: not enough non-rep old to remove (32) | new补充=174[W06] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=34651, new=357 | pseudo_fill=217[W07] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34595, new=402 | pseudo_fill=273[W08] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34696, new=340 | pseudo_fill=172[W09] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34775, new=236 | pseudo_fill=93[W10] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34877, new=30 | WARN: not enough non-rep old to remove (191) | new补充=170[W11] stable | lwf=0.50 ep=180 | buf=35068
[After CL]
  Acc=0.9046  Pre=0.9165  Rec=0.9096  F1=0.9131
  Confusion: TP=40821 FP=3719 FN=4055 TN=32892
Random seed set to: 5012

============================================================
Seed 5012 (2/5) | dataset=unsw | mode=ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8186  Pre=0.7566  Rec=0.9887  F1=0.8572
  Confusion: TP=44819 FP=14421 FN=513 TN=22579
  Samples: old=35032, new=30 | WARN: not enough non-rep old to remove (36) | new补充=170[W00] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35004, new=52 | WARN: not enough non-rep old to remove (64) | new补充=148[W01] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35031, new=41 | WARN: not enough non-rep old to remove (37) | new补充=159[W02] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35015, new=33 | WARN: not enough non-rep old to remove (53) | new补充=167[W03] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35040, new=18 | WARN: not enough non-rep old to remove (28) | new补充=182[W04] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35041, new=13 | WARN: not enough non-rep old to remove (27) | new补充=187[W05] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35025, new=31 | WARN: not enough non-rep old to remove (43) | new补充=169[W06] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=34714, new=331 | pseudo_fill=154[W07] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34641, new=363 | pseudo_fill=227[W08] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34684, new=300 | pseudo_fill=184[W09] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34791, new=212 | pseudo_fill=77[W10] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34790, new=52 | new补充=148[W11] stable | lwf=0.50 ep=180 | buf=35068
[After CL]
  Acc=0.8958  Pre=0.8818  Rec=0.9362  F1=0.9082
  Confusion: TP=42015 FP=5631 FN=2865 TN=30999
Random seed set to: 5013

============================================================
Seed 5013 (3/5) | dataset=unsw | mode=ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8178  Pre=0.7553  Rec=0.9897  F1=0.8568
  Confusion: TP=44866 FP=14535 FN=466 TN=22465
  Samples: old=35013, new=46 | WARN: not enough non-rep old to remove (55) | new补充=154[W00] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35026, new=49 | WARN: not enough non-rep old to remove (42) | new补充=151[W01] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35018, new=50 | WARN: not enough non-rep old to remove (50) | new补充=150[W02] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35004, new=36 | WARN: not enough non-rep old to remove (64) | new补充=164[W03] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35040, new=23 | WARN: not enough non-rep old to remove (28) | new补充=177[W04] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35027, new=22 | WARN: not enough non-rep old to remove (41) | new补充=178[W05] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35041, new=16 | WARN: not enough non-rep old to remove (27) | new补充=184[W06] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=34701, new=355 | pseudo_fill=167[W07] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34601, new=436 | pseudo_fill=267[W08] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34721, new=315 | pseudo_fill=147[W09] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34770, new=254 | pseudo_fill=98[W10] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34898, new=36 | WARN: not enough non-rep old to remove (170) | new补充=164[W11] stable | lwf=0.50 ep=180 | buf=35068
[After CL]
  Acc=0.8978  Pre=0.8714  Rec=0.9555  F1=0.9115
  Confusion: TP=42892 FP=6330 FN=1999 TN=30272
Random seed set to: 5014

============================================================
Seed 5014 (4/5) | dataset=unsw | mode=ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8168  Pre=0.7528  Rec=0.9934  F1=0.8565
  Confusion: TP=45034 FP=14788 FN=298 TN=22212
  Samples: old=35026, new=38 | WARN: not enough non-rep old (42) | new补充=162[W00] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=35001, new=45 | WARN: not enough non-rep old to remove (67) | new补充=155[W01] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35007, new=46 | WARN: not enough non-rep old to remove (61) | new补充=154[W02] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35037, new=31 | WARN: not enough non-rep old to remove (31) | new补充=169[W03] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35031, new=28 | WARN: not enough non-rep old to remove (37) | new补充=172[W04] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35036, new=24 | WARN: not enough non-rep old to remove (32) | new补充=176[W05] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35030, new=27 | WARN: not enough non-rep old to remove (38) | new补充=173[W06] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=34695, new=324 | pseudo_fill=173[W07] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34638, new=386 | pseudo_fill=230[W08] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34867, new=158 | new补充=42 | pseudo_fill=1[W09] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34823, new=192 | new补充=8 | pseudo_fill=45[W10] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34862, new=38 | new补充=162[W11] stable | lwf=0.50 ep=180 | buf=35068
[After CL]
  Acc=0.8946  Pre=0.8638  Rec=0.9599  F1=0.9093
  Confusion: TP=43070 FP=6790 FN=1798 TN=29844
Random seed set to: 5015

============================================================
Seed 5015 (5/5) | dataset=unsw | mode=ssf
============================================================
[Offline] Training 200 epochs... done
[Before CL]
  Acc=0.8183  Pre=0.7557  Rec=0.9901  F1=0.8572
  Confusion: TP=44884 FP=14510 FN=448 TN=22490
  Samples: old=35010, new=29 | WARN: not enough non-rep old to remove (58) | new补充=171[W00] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35015, new=29 | WARN: not enough non-rep old to remove (53) | new补充=171[W01] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35035, new=26 | WARN: not enough non-rep old to remove (33) | new补充=174[W02] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35026, new=33 | WARN: not enough non-rep old to remove (42) | new补充=167[W03] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35035, new=28 | WARN: not enough non-rep old to remove (33) | new补充=172[W04] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=35009, new=48 | WARN: not enough non-rep old to remove (59) | new补充=152[W05] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=34911, new=134 | WARN: not enough non-rep old to remove (157) | new补充=66[W06] stable | lwf=0.50 ep=180 | buf=35068
  Samples: old=34745, new=298 | pseudo_fill=123[W07] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34654, new=415 | pseudo_fill=214[W08] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34829, new=202 | pseudo_fill=39[W09] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34828, new=192 | new补充=8 | pseudo_fill=40[W10] DRIFT | lwf=0.00 ep=180 | buf=35068
  Samples: old=34833, new=46 | new补充=154[W11] stable | lwf=0.50 ep=180 | buf=35068
[After CL]
  Acc=0.8893  Pre=0.8490  Rec=0.9718  F1=0.9062
  Confusion: TP=43629 FP=7760 FN=1267 TN=28859
