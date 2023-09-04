from easydict import EasyDict as edict

config = edict()

config.path = edict()
config.path.data = "./data/"
config.path.model = "./model/"

config.train = edict()
config.train.epoch = 25
config.train.lr = 1e-1