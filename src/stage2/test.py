import torch, os
import torch.nn as nn
import fairseq  
from torch.utils.data import DataLoader
from stage2MOS import MosPredictor, MyDataset, sample
import numpy as np
import scipy.stats
import json

ref = os.environ.get("REF")


## 1. load in pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cp_path = f'{ref}/table1_data/wav2vec_small.pt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
ssl_model = model[0]
ssl_model.remove_pretraining_modules()

## running this on CPU-only generally does not work
model = MosPredictor(ssl_model).to(device)
model.eval()

import os
print(os.getcwd())
print('Loading checkpoint')

metrics = {
    'MSE': [],
    'LCC': [],
    'SRCC': [],
    'KTau': [],
    'average': []
    }

for ck in range(3):
    print(f"Trying ckpt {ck}-----\n")
    my_checkpoint = f"checkpoints/my_MOS_SRCC_{ck}"  #####
    model.load_state_dict(torch.load(my_checkpoint))

    print('Loading data')
    wavdir = f'{ref}/table1_data/stage2MOS/DATA/wav'
    validlist = f'{ref}/table1_data/stage2MOS/DATA/sets/test_mos_list.txt'
    validset = MyDataset(wavdir, validlist, val=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=1, collate_fn=validset.collate_fn)

    total_loss = 0.0
    num_steps = 0.0
    predictions = { }  # filename : prediction
    #criterion = nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    print('Starting prediction')
    for i, data in enumerate(validloader, 0):
        inputs, labels, filenames, _ = data
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Features extraction
        inputs = inputs.squeeze(1)  ## [batches, audio_len]
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        output = outputs.cpu().detach().numpy()[0]
        #print(output_vec)
        #classIdx = output_vec.index(max(output_vec))
        #classPred = classes[classIdx]
        #predictions[filenames[0]] = classPred  ## batch size = 1
        predictions[filenames[0]] = output

    ## compute correlations [utterance level]
    ## load in true labels
    true_MOS = { }
    validf = open(validlist, 'r')
    for line in validf:
        parts = line.strip().split(',')
        uttID = parts[0]
        MOS = float(parts[1])
        true_MOS[uttID] = MOS

    ## compute correls.
    sorted_uttIDs = sorted(predictions.keys())
    ts = []
    ps = []
    for uttID in sorted_uttIDs:
        t = true_MOS[uttID]
        p = predictions[uttID]
        ts.append(t)
        ps.append(p)

    truths = np.array(ts)
    preds = np.array(ps)
        
    ### UTTERANCE
    MSE=np.mean((truths-preds)**2)
    print('[UTTERANCE] Test error= %f' % MSE)
    LCC=np.corrcoef(truths, preds)
    print('[UTTERANCE] Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(truths.T, preds.T)
    print('[UTTERANCE] Spearman rank correlation coefficient= %f' % SRCC[0])
    KTAU=scipy.stats.kendalltau(truths, preds)
    print('[UTTERANCE] Kendall Tau rank correlation coefficient= %f' % KTAU[0])

    metrics['MSE'].append(MSE)
    metrics['LCC'].append(LCC[0][1])
    metrics['SRCC'].append(SRCC[0])
    metrics['KTau'].append(KTAU[0])


metrics['average'].append(np.mean(metrics['MSE']))
metrics['average'].append(np.mean(metrics['LCC']))
metrics['average'].append(np.mean(metrics['SRCC']))
metrics['average'].append(np.mean(metrics['KTau']))


print(metrics['average'])
