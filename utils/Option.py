class param(object):
    def __init__(self):
        # Path
        self.ROOT = 'C:/Users/rlawj/WORK/SIDE_PROJECT/DACON/BLOCK_CLASSIFICATION'
        self.DATASET_PATH = f'{self.ROOT}/DB/'
        self.OUTPUT_CKP = f'{self.ROOT}/backup/try2/ckp_aug_fine_tuning'
        self.OUTPUT_LOG = f'{self.ROOT}/backup/try2/log_aug_fine_tuning'
        self.CKP_LOAD = True

        # Data
        self.SIZE = 224

        # Train or Test
        self.EPOCH = 5
        self.LR = 1e-4
        self.B1 = 0.5
        self.B2 = 0.999
        self.BATCHSZ = 8

        # Handler
        # run_type 0 : train, 1 : test
        self.run_type = 1