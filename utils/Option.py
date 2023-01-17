class param(object):
    def __init__(self):
        # Path
        self.ROOT = ''
        self.DATASET_PATH = f'{self.ROOT}/DB/'
        self.OUTPUT_CKP = f'{self.ROOT}/backup/ckp'
        self.OUTPUT_LOG = f'{self.ROOT}/backup/log'
        self.CKP_LOAD = True

        # Data
        self.SIZE = 224

        # Train or Test
        self.EPOCH = 5
        self.LR = 1e-4
        self.BATCHSZ = 8

        # Handler
        # run_type 0 : train, 1 : test
        self.run_type = 1
