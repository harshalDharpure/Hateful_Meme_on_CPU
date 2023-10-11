import torch
import numpy as np
import random

import config
import os
from train import train_for_epoch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

# ... (rest of the imports)

import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    opt = config.parse_opt()
    set_seed(opt.SEED)

    # Create tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    constructor = 'build_baseline'

    if opt.MODEL == 'pbm':
        from dataset import Multimodal_Data
        import baseline
        train_set = Multimodal_Data(opt, tokenizer, opt.DATASET, 'train', opt.SEED - 1111)
        test_set = Multimodal_Data(opt, tokenizer, opt.DATASET, 'test')
        label_list = [train_set.label_mapping_id[i] for i in train_set.label_mapping_word.keys()]
        model = getattr(baseline, constructor)(opt, label_list)
    else:
        from roberta_dataset import Roberta_Data
        import roberta_baseline
        train_set = Roberta_Data(opt, tokenizer, opt.DATASET, 'train', opt.SEED - 1111)
        test_set = Roberta_Data(opt, tokenizer, opt.DATASET, 'test')
        model = getattr(roberta_baseline, constructor)(opt)

    # Load only the first 500 samples
    train_set = torch.utils.data.Subset(train_set, range(500))
    test_set = torch.utils.data.Subset(test_set, range(500))

    # Adjust the batch size if needed (make sure it's <= 500)
    batch_size = min(opt.BATCH_SIZE, len(train_set))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)
    train_for_epoch(opt, model, train_loader, test_loader)

    exit(0)
