import os
import fairseq
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import random
from tqdm.auto import tqdm
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


class MosPredictor(nn.Module):
    def __init__(self,ssl_model):
        
        super(MosPredictor, self).__init__()
        
        self.ssl_model = ssl_model
        self.ssl_features = 768 

        #Linear layer
        self.output_layer = nn.Linear(self.ssl_features, 1)
        
    def forward(self, inputs):
        '''
        Forward pass of the MosPredictor module.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, sequence_length, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size].
        '''
        
        res = self.ssl_model(inputs, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)
        
        # Linear layer
        x = self.output_layer(x)
        
        return x.squeeze(1)

def ERLDM(p,g,type='abs', factor_=0.1): 

    '''
    E-RLDM loss function.

    Args:
        p (torch.Tensor): Predicted values tensor.
        g (torch.Tensor): Ground truth values tensor.
        type (str): Loss type. 'abs' for absolute difference loss, 'mse' for mean squared error loss.
        factor (float): Loss factor if the relative order is preserved.

    Returns:
        torch.Tensor: Calculated loss value.
    '''

    device = p.device
    p_old = torch.cat(P).clone().detach()
    g_old = torch.cat(G).clone().detach()

    total = (p_old.shape[0]*len(p)) - len(p) + 1
    
    neg = p_old.shape[0]
    g_ = g.view(-1,1).repeat(1,neg).view(-1) - g_old.repeat(len(g))
    p_ = p.view(-1,1).repeat(1,neg).view(-1) - p_old.repeat(len(p))

    factor = g_.clone().detach()*p_.clone().detach()
    factor = torch.tensor([factor_  if i>=0 else 1.0 for i in factor]).to(device)
    loss = 0
  
    if type == 'abs': 
        loss += (torch.abs(g_ - p_)*factor).sum()
    if type == 'mse': 
        loss += (((g_ - p_)**2)*factor).sum()
    loss /= total

    return loss

def RLDM(p,g,type='abs',extend=False, batch_size=32, extend_num=1024, factor=0.1):
    '''
    Relative location dissimlarity minmization loss function.

    Args:
        p (torch.Tensor): Predicted values tensor.
        g (torch.Tensor): Ground truth values tensor.
        type (str): Loss type. 'abs' for absolute difference loss, 'mse' for mean squared error loss.
        extend (bool): Flag indicating whether to include E-RLDM loss in the total loss calculation.
        batch_size (int): Batch size.
        extend_num (int): Number of extended samples.
        factor (float): Loss factor if the relative order is preserved.

    Returns:
        torch.Tensor: Calculated loss value.
    '''

    total = 1
    loss = 0.0

    if len(p) > 1: 
        p = (p - p.min() ) / ((p.max() - p.min()) +1e-12)
        g = (g - g.min() ) / ((g.max() - g.min()) + 1e-12)

        total = (len(p)*len(p)) - len(p)
        for r,i in enumerate(zip(g,p)):
            for c,j in enumerate(zip(g,p)):
                g_ = i[0]-j[0]
                p_ = i[1]-j[1]
    
                if type == 'abs':
                    # UTMOS paper loss
                    # loss += torch.max( torch.tensor(0.0).to(device), torch.abs(g_ - p_) - 0.5 )
                    if g_*p_ >= 0: loss += torch.abs(g_ - p_)*factor
                    else: loss += torch.abs(g_ - p_)
                if type == 'mse':
                    if g_*p_ >= 0: loss += ((g_ - p_)**2)*factor
                    else: loss += (g_ - p_)**2
    else: 
        for g_,p_ in zip(g,p):
            if type == 'abs': loss += torch.abs(g_ - p_)
            if type == 'mse': loss += (g_ - p_)**2

    loss /= total

    if extend:
        if len(G) != 0: 
            loss += ERLDM(p,g,type,factor)
   
        with torch.no_grad():
            
            if len(G)*batch_size > extend_num: 
                G.pop(0)
                P.pop(0)

            P.append(p.clone().detach())
            G.append(g.clone().detach())
    
    return loss
  
class MyDataset(Dataset):
    def __init__(self, wavdir, mos_list,split='l',val=False):
        '''
        Custom dataset class for loading waveform data and MOS scores.

        Args:
            wavdir (str): Directory path containing the waveform files.
            mos_list (str): Path to the file containing teh file names and theri MOS scores.
            split (str): Split type. 'l' for labeled, 'ul' for unlabeled.
            val (bool): Flag indicating whether the dataset is validation or not.

        Attributes:
            mos_lookup (dict): Dictionary to store MOS scores for each waveform.
            wavdir (str): Directory path containing the waveform files.
            wavnames (list): List of sorted waveform names.
            labelled (list): List corresponding to waveform names -- if labelled True else False .
        '''

        self.mos_lookup = { }
        f = open(mos_list, 'r')
        labelled = []
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            mos = float(parts[1])
            if val: 
                self.mos_lookup[wavname] = [mos,True]
            elif parts[2] == 'l': 
                self.mos_lookup[wavname] = [mos,True]
            elif parts[2] == 'ul': 
                self.mos_lookup[wavname] = [mos,False]
            else: 
                print("--Something is wrong--")
            # self.mos_lookup[wavname] = mos
        
        #                 '4.75', '4.875', '5.0' ]
        self.wavdir = wavdir
        self.wavnames = sorted(self.mos_lookup.keys()) 
        
       
        self.wavnames_l = [k for k,v in self.mos_lookup.items() if v[1]]
        self.mos_lookup_l = {i:self.mos_lookup[i] for i in self.wavnames_l}
       
        self.wavnames_ul = [k for k,v in self.mos_lookup.items() if not v[1]]
        self.mos_lookup_ul = {i:self.mos_lookup[i] for i in self.wavnames_ul}

        print(f"Labelled samples are -- {len(self.mos_lookup_l)}")
        print(f"UnLabelled samples are -- {len(self.mos_lookup_ul)}")
        
        self.labelled = []
        if split == 'l':
            self.wavnames = self.wavnames_l
            self.mos_lookup = self.mos_lookup_l
            self.labelled = [True]*len(self.wavnames)
        elif split == 'ul':
            self.wavnames = self.wavnames_ul
            self.mos_lookup = self.mos_lookup_ul 
            self.labelled = [True]*len(self.wavnames)
        elif split == 'both':
            self.wavnames = self.wavnames_l + self.wavnames_ul
            self.mos_lookup = {**self.mos_lookup_l,**self.mos_lookup_ul}
            self.labelled = [True]*len(self.wavnames_l) + [False]*len(self.wavnames_ul)
        
        # print(self.mos_lookup)

        
    def __getitem__(self, idx):
        '''
        Retrieves the waveform, MOS score, and waveform name at the given index.

        Args:
            idx (int): Index of the data sample to retrieve.

        Returns:
            tuple: Tuple containing the waveform, MOS score, waveform name, and labelled or unlabeled.
        '''
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
     
        score = self.mos_lookup[wavname][0]

        return wav, score, wavname, self.mos_lookup[wavname][1]
    
    def __len__(self):
        '''
        Returns the total number of data samples in the dataset.

        Returns:
            int: Total number of data samples.
        '''
         
        return len(self.wavnames)

    def collate_fn(self, batch):  ## make them all the same length with zero padding
        '''
        Collate function for creating mini-batches.

        Args:
            batch (list): List of tuples, where each tuple contains the waveform, MOS score, and waveform name.

        Returns:
            tuple: Tuple containing the padded waveforms, MOS scores, and waveform names.
        '''
        wavs, scores, wavnames, labelled = zip(*batch)
        wavs = [i[:,:51120] for i in list(wavs)]
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []

        for wav in wavs:
        
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)

        output_wavs = torch.stack(output_wavs, dim=0)
        scores  = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        return output_wavs, scores, wavnames, labelled
    
def sample(mos_list,label,labelled_count=0,unlabelled_count=0): 
    '''

    Function to sample the dataset into labelled and unlabelled sets used for stage 2 finetuning.
    Args:
        mos_list (str): Path to the file containing the file names and theri MOS scores.
        label (str): Name of the file to be saved.
        labelled_count (int): Number of labelled samples to be sampled.
        unlabelled_count (int): To use the remaining unlablled samples or not.
    
    Returns:
        None

    '''
    mos_lookup = []
    f = open(mos_list, 'r')
    for line in f:
        parts = line.strip().split(',')
        wavname = parts[0]
        mos = float(parts[1])
        mos_lookup.append([wavname,mos])
    
    labelled = random.sample(range(len(mos_lookup)),labelled_count)
    print("Sampled indexes are: ---",labelled)

    a =  ""
    for i,j in enumerate(mos_lookup):
        if i in labelled:
            print("---",i,j)
            a += f"{j[0]},{j[1]},l\n"
        elif unlabelled_count != 0: a += f"{j[0]},0,ul\n"
    
    with open(f"{label}.txt", "w") as f:
        f.write(a)

if __name__ == '__main__':
    
    ref = os.environ.get("REF")

    random.seed(1984)

    batch_size=8  ## 16 was OOM on CUDA.  need to use 2 for validation
    type_ = 'abs'
    extend_num = 5000
    factor=0.1
    labelled_count = 10 
    unlabeled_count = 1
    orig_patience=100
    patience = orig_patience
    bapmos = True
    num_bins = 10
    
    print("-"*10)
    print(f"Loss type: {type_}")
    print(f"Batch size: {batch_size}")
    print(f"Extend num: {extend_num}")
    print(f"Factor: {factor}") 
    print(f"Patience: {patience}")
    print(f"Random seed value: 1984")
    print(f"Labelled count: {labelled_count}")
    print(f"Unlabelled count: {unlabeled_count}")
    print(f"Use BapMOS: {bapmos}")
    print(f"Number of bins: {num_bins}")
    print("-"*10)

    P,G = [], [] # Lists to add the extended number of samples for the E-RLDM loss. 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    # path to the wav2vec model
    cp_path = f'{ref}/table1_data/wav2vec_small.pt'

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    net = MosPredictor(ssl_model)
    net = net.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.95)

    
    wavdir = f'{ref}/table1_data/stage2MOS/DATA/wav'
    trainlist = f'{ref}/table1_data/stage2MOS/DATA/sets/train_mos_list.txt'
    validlist = f'{ref}/table1_data/stage2MOS/DATA/sets/val_mos_list.txt'

    sample(trainlist,label="train", labelled_count=labelled_count, unlabelled_count=unlabeled_count)
    trainlist = 'train.txt'

    try: os.mkdir("checkpoints")
    except: pass

    for run in range(3):
        # This is tricky to set, choose stage1 if number of labelled samples is 0 (zero) or choose stage 2 fintuned model on the lablled samples only i.e., 0 unlablled samples. For stage 2 since we do do 3 runs, having different labelled samples (random sampling), we use the pretrained weights of the same run i.e., same labelled samples.
        print(f"Set the pretrained weighte manually for the run {run}") 
        my_checkpoint = f""
        # my_checkpoint = f"{ref}/pretrainedweights/stage1/prestage1" # this is for 0 lablled samples case i.e., stage1 MOS trained model weights  
        # load the stage1 MOS trained model weights or a stage2 finetuned model on the labelled set only.
        net.load_state_dict(torch.load(my_checkpoint)) 
        
        PREV_SRCC = -10
        patience = orig_patience
        
        name_score = {}
        names_to_use = []

        validset = MyDataset(wavdir, validlist, val=True)
        validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=4, collate_fn=validset.collate_fn)

        # Initialze the optimizer with a very small learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.000001

        print("---Starting training---")
        for epoch in range(0,1001):
        
            trainset = MyDataset(wavdir, trainlist,split='both')
            print(f"Number of samples with the pseudo scores are-: {len(name_score)}")
            # Load the trainset with the pseudo scores
            for k,v in name_score.items():
                trainset.mos_lookup[k] = [v,False]
            # Gets activated when using BapMOS.
            if len(names_to_use) != 0: 
                trainset.wavnames = names_to_use       
                print(f"Number of samples with the pseudo scores used for training are-: {len(names_to_use)}")
            trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4, collate_fn=trainset.collate_fn)
        
            P,G = [], []
            semi = 50
            for param_group in optimizer.param_groups:
                if epoch < semi:
                    param_group['lr'] *= 1.1
                if epoch == semi:
                    param_group['lr'] = 0.001
                if epoch > semi:
                    param_group['lr'] *= 0.99

            print(f"Current learning rate: {param_group['lr']}")

            STEPS=0
            net.train()
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                if epoch == 0: break
                optimizer.zero_grad()

                inputs, labels, filenames, labelled = data
                inputs = inputs.to(device)
                inputs = inputs.squeeze(1)
                labels = labels.to(device)

                loss = torch.tensor(0.0).to(device)            
                # feature extraction
                outputs = net(inputs)
                
                # labelled output
                output_l = outputs[torch.tensor(labelled)]
                # lablled labels
                labels_l = labels[torch.tensor(labelled)]
                # labelled loss
                if len(output_l) > 0: 
                    loss += RLDM(output_l,labels_l,type=type_,extend=True, batch_size=batch_size, extend_num=extend_num, factor=factor)
                # print(loss, len(labels_l), "-l")

                # unlabelled output
                output_ul = outputs[torch.logical_not(torch.tensor(labelled))]
                # unlabelled loss  
                if len(output_ul) > 0: 
                    # unlablled labels
                    labels_ul = labels[torch.logical_not(torch.tensor(labelled))]
                    loss += RLDM(output_ul,labels_ul,type=type_,extend=True, batch_size=batch_size, extend_num=extend_num,factor=0.0)
                # print(loss, len(labels_ul), "-ul")
                loss.backward()
                optimizer.step()
                STEPS += 1
                running_loss += loss.item()
            # exit()
            STEPS += 1  
            print('EPOCH: ' + str(epoch))
            print('AVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS))

            ## have it run validation every epoch & print loss
            epoch_val_loss = 0.0
            net.eval()
            ## clear memory to avoid OOM
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

            ## validation
            VALSTEPS=0
            predictions = { }  # filename : prediction

            for i, data in enumerate(validloader, 0):
                VALSTEPS+=1
                inputs, labels, filenames, _ = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Features extraction
                inputs = inputs.squeeze(1)  ## [batches, audio_len]
                outputs = net(inputs)
                
                loss = criterion(outputs, labels)
                # loss = RLDM(outputs,labels,type=type_) 
                epoch_val_loss += loss.item()

                output = outputs.cpu().detach().numpy()[0]
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
            print('---[UTTERANCE] Test error= %f' % MSE)
            LCC=np.corrcoef(truths, preds)
            print('---[UTTERANCE] Linear correlation coefficient= %f' % LCC[0][1])
            SRCC=scipy.stats.spearmanr(truths.T, preds.T)
            print('---[UTTERANCE] Spearman rank correlation coefficient= %f' % SRCC[0])
            KTAU=scipy.stats.kendalltau(truths, preds)
            print('---[UTTERANCE] Kendall Tau rank correlation coefficient= %f' % KTAU[0])

            if SRCC[0] > PREV_SRCC:
                print('SRCC has decreased')
                PREV_SRCC=SRCC[0]
                PATH = './checkpoints/my_MOS_SRCC_' + str(run) 
                torch.save(net.state_dict(), PATH)
                patience = orig_patience  ## reset  

                # semi supervised data creation        
                ulset = MyDataset(wavdir, trainlist, split='ul')
                ulloader = DataLoader(ulset, batch_size=1, shuffle=False, num_workers=4, collate_fn=trainset.collate_fn)
                name_score = {}
                bin = []

                with torch.no_grad():
                    for i, data in enumerate(ulloader, 0):
                        inputs, labels, filenames, labelled = data
                        inputs = inputs.to(device)
                        inputs = inputs.squeeze(1)
                        labels = labels.to(device)

                        # feature extraction
                        output_ul = net(inputs)
                        bin.append(output_ul.item())

                        for i_ in range(len(filenames)):
                            name_score[filenames[i_]] = output_ul[i_].item()
                    print("finished extraction")

                # BapMOS code
                if bapmos:
                    print("Runnins BapMOS")
                    # Calculate the histogram
                    counts, bins, _ = plt.hist(bin, bins=num_bins)  # Adjust the number of bins as per your preference

                    try: os.mkdir("bins")
                    except: pass
                    # Plotting the histogram after each time we update the pseudo MOS values.
                    plt.hist(bin, bins=num_bins)  
                    plt.xlabel('Bins')
                    plt.ylabel('Frequency')
                    plt.title('Histogram of Bins')  
                    plt.savefig(f'bins/{epoch}_hist.png')
                    plt.clf()

                    bins = [[bins[i], bins[i+1], 0] for i in range(len(bins)-1)]


                    # Shuffle the dictionary
                    def shuffle_dictionary(dictionary):
                        keys = list(dictionary.keys())
                        random.shuffle(keys)
                        shuffled_dict = {}
                        for key in keys:
                            shuffled_dict[key] = dictionary[key]
                        return shuffled_dict

                    names_to_use = []
                    for names,scores in shuffle_dictionary(name_score).items():
                        for i in range(len(bins)):
                            if bins[i][0] <= scores <= bins[i][1]:
                                bins[i][2] += 1
                                if bins[i][2] <= min(counts): 
                                    names_to_use.append(names)
                    print(f"count of files in each bin: {counts}")


            else:
                pass
                # patience-=1
                # if patience == 0:
                #     print('SRCC has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                #     break
            
            avg_val_loss=epoch_val_loss/VALSTEPS    
            print('EPOCH VAL LOSS: ' + str(avg_val_loss))

    print('Finished Training')

