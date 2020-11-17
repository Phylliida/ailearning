from importlib import reload
import minGPTOther
from minGPTOther import mingpt
from minGPTOther.mingpt.model import GPT, GPTConfig
from minGPTOther.mingpt.trainer import Trainer, TrainerConfig
reload(minGPTOther.mingpt.trainer)
reload(minGPTOther.mingpt.model)
from minGPTOther.mingpt.model import GPT, GPTConfig
from minGPTOther.mingpt.trainer import Trainer, TrainerConfig

global model # so we can access the model even if we cancel the run partways through
global trainer
import torch


def trainTransformer(prevModel, train_dataset, test_dataset, n_layer, n_head, n_embd, lr, trainCallback=None, showProgress=True, loadFromCheckpoint=False, useCuda=True):
    global model
    global trainer
    
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, 
                      n_layer=n_layer, n_head=n_head, n_embd=n_embd, show_progress=showProgress)
    if prevModel is None:
        model = GPT(mconf)
    else:
        model = prevModel
    saveString = f'layer{mconf.n_layer}head{mconf.n_head}emb{mconf.n_embd}{str(type(train_dataset).__name__)}{str(train_dataset)}'
    lossesPath = saveString + "losses.json"
    tconf = TrainerConfig(max_epochs=10000, batch_size=1024, learning_rate=lr,
                          lr_decay=True, warmup_tokens=0, final_tokens=len(train_dataset)*10,
                          num_workers=0, ckpt_path=saveString, losses_path=lossesPath, show_progress=showProgress)
    trainer = Trainer(model, train_dataset, test_dataset, tconf, useCuda=useCuda, loadFromCheckpoint=loadFromCheckpoint)
    trainer.train(trainCallback)
