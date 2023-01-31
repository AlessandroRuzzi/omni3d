from .datasets.dataloader import CreateDataLoader
from .datasets.train_options import TrainOptions
import wandb


opt = TrainOptions().parse()
train_dl, val_dl, test_dl = CreateDataLoader(opt)
train_ds, test_ds = train_dl.dataset, test_dl.dataset

val_ds = val_dl.dataset if val_dl is not None else None

print(len(train_ds),len(val_ds),len(test_ds),)