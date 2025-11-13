from utils import load_data_memmap, data_forms
import numpy as np
# load babylm dataset
class BaseDataLoader():
    def __init__(self,size,eot_token_id=None):
        self.eot_token_id = eot_token_id
        self.size = size
        self.data_forms = data_forms
    
    def get_batch(self,split='train',block_size=256,
                                        batch_size=32):
        total_len= (block_size * batch_size)+1
        combo = [load_data_memmap(size=self.size,form=data_form,
                                                    split=split)
        for data_form in self.data_forms]
        combo = np.concatenate(combo,axis=0)
        dl = len(combo)
        ix = np.random.randint(0,dl-total_len-1)
        combo = combo[ix:ix+total_len]
        #print('Total len;',total_len, len(id_bucket))
        id_bucket = combo[:-1]
        y_bucket = combo.copy()[1:]
        #print('Total len;',total_len,len(id_bucket),len(y_bucket))
        
        return id_bucket.astype(np.int64).reshape(batch_size,block_size), y_bucket.astype(np.int64).reshape(batch_size,block_size)
        

    
if __name__ == '__main__':
    dataloader = BaseDataLoader('10M',data_form='bnc_spoken')#

    for i in range(50): 
        data1 = dataloader.get_batch(split='dev')