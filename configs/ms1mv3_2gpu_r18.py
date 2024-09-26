from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r18"
config.resume = False
config.pretrained = "./pretrained/model.pt"
config.output = "./output"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.004 #0.02
config.verbose = 2000
config.dali = False
config.backbonefrz = True

config.rec = "/home/diwu/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 4 #30
config.warmup_epoch = 0
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
