import os
ROOT = '/content/'
DRIVE_ROOT = os.path.join('content','drive','MyDrive','simplicity')
RU_ADAPT_PATHS = [os.path.join(ROOT, 'RuAdapt', 'Adapted_literature', 'zlatoust_sentence_aligned_with_CATS.csv'),
     os.path.join(ROOT, 'RuAdapt', 'Encyclopedic', 'lsslovar_B_to_A_sent.csv'),
     os.path.join(ROOT, 'RuAdapt', 'Encyclopedic', 'lsslovar_C_to_A_sent.csv'),
     os.path.join(ROOT, 'RuAdapt', 'Encyclopedic', 'lsslovar_C_to_B_sent.csv'),
     os.path.join(ROOT, 'RuAdapt', 'Fairytales', 'df_fairytales_sent.csv')]
RUSIMPLESENTEVAL_PATH = os.path.join(ROOT, 'RuSimpleSentEval','dev_sents.csv')
TEST_DATA_PATH = os.path.join(ROOT, 'RuSimpleSentEval','public_test_sents.csv')
LOG_PATH = os.path.join(DRIVE_ROOT, 'train.logs')
LOG_PATH_LOCAL = '/content/train.logs'