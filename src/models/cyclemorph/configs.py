import ml_collections
from functools import partial
import torch.nn as nn

def get_CycleMorphV0_config():
    config = ml_collections.ConfigDict()
    config.batchSize = 1
    config.inputSize = (64, 64, 64)
    config.fineSize = (16, 16, 16)
    config.input_nc = 2
    config.encoder_nc = (16, 32, 32, 32, 32)
    config.decoder_nc = (32, 32, 32, 8, 8, 3)
    config.beta1 = 0.005
    config.lr = 0.0001
    config.lambda_A = 0.0001
    config.lambda_B = 0.0004
    config.lambda_R = 1
    config.lr_policy = 'lambda'
    config.lr_decay_iters = 50
    config.isTrain = True
    config.gpu_ids = [0]
    config.which_model_net = 'registUnet'
    config.init_type = 'normal'
    config.continue_train = False
    config.epoch_count = 0
    config.niter = 100
    config.niter_decay = 100
    return config