Simply run the  stage2MOS.py and stage2MOSBapMOS.py for training on the dataset used in the paper.  

Run test.py to evaluate the trained model.


## NOTES
* Set labelled_count to the ones used in paper or your own choice. 
* Set unlabelled_count as 0 if you do not want to use any unlabelled samples or any other other number to use the remaining samples in the data as unlabelled.
* What pre-trained weights to use for stage 2 at this location in the script: https://github.com/nii-yamagishilab-visitors/partial_rank_similarity/blob/8a6616c25601e9a8edeacd10d0c2f500e2046687/src/stage2/stage2MOS.py#L355. This is tricky to set, choose stage1 if number of labelled samples is 0 (zero) or choose stage 2 fintuned model on the lablled samples only i.e., 0 unlablled samples. For stage 2 since we do do 3 runs, having different labelled samples (random sampling), we use the pretrained weights of the same run i.e., same labelled samples.
