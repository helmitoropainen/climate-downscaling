##############################
# Model training and loading # 
# using PyTorch Lightning    #
##############################

import os
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch.optim as optim
import torch
import glob

import network
from utils import get_loss_flow, get_loss_diffusion, is_nordics
from evaluation import evaluate_model

# define the LightningModule
class Model(L.LightningModule):
    def __init__(self, args, train_shape_out):
        super().__init__()
        self.lr = args.lr
        self.model_type = args.model
        self.bias_flow = args.bias_flow
        self.nordics = True if is_nordics(args) else False
        label_dim = 0
        if self.model_type == "diffusion": 
            network_class = network.EDM
            self.loss = get_loss_diffusion
            in_dim = args.dim_channels * 2 
        else: 
            network_class = network.Flow
            self.loss = get_loss_flow
            if args.nordics_constants == "True":
                n_conts = 3 # t, land-cover, elevation (if embed t, one channel is substracted) 
                if args.nordics_time == "True":
                    n_conts += 2 # 2 previous timesteps as additional condition
                in_dim = args.dim_channels + n_conts
                label_dim = 3 # row, col, doy
            else:
                in_dim = args.dim_channels + 1

        self.model = network_class(
            img_resolution=(train_shape_out[-2], train_shape_out[-1]), 
            in_channels=in_dim, 
            out_channels=args.dim_channels, 
            label_dim=label_dim, 
            model_channels=args.number_channels, 
            num_blocks=args.number_residual_blocks,) 
        
    def training_step(self, batch):
        if self.nordics:
            inputs = batch["inputs"]       
            targets = batch["targets"]
            labels = batch.get("labels", None)
            aux = batch.get("aux", None)
        else:
            inputs, targets = batch
            labels = None
            aux = None

        if self.model_type == "flow": loss = self.loss(self.model, targets, inputs, bias=self.bias_flow, nordics=self.nordics, labels=labels, aux=aux)
        else: loss = self.loss(self.model, targets, inputs)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        if self.nordics:
            inputs = batch["inputs"]       
            targets = batch["targets"]
            labels = batch.get("labels", None)
            aux = batch.get("aux", None)
        else:
            inputs, targets = batch
            labels = None
            aux = None

        if self.model_type == "flow": loss = self.loss(self.model, targets, inputs, bias=self.bias_flow, nordics=self.nordics, labels=labels, aux=aux)
        else: loss = self.loss(self.model, targets, inputs)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# training setup
def run_training(args, data, resume=False):
    train_data = data["train"]
    if args.use_validation == "yes": val_data = data["val"]
    else: val_data = None
    train_shape = data["train_shape_out"]

    # model
    model = Model(args, train_shape)

    # checkpoints
    checkpoint_callback_validation = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.model_path,
        filename = f'{args.model_id}-{{epoch:02d}}'
    )

    checkpoint_callback_all = ModelCheckpoint(
        monitor=None,
        save_top_k = -1,
        dirpath=args.model_path,
        filename = f'{args.model_id}-{{epoch:02d}}'
    )

    if args.use_validation == "yes": callbacks=[checkpoint_callback_validation] 
    else: callbacks=[checkpoint_callback_all]

    # tensorboard
    logger = TensorBoardLogger("tb_logs", name=f"{args.model}_model")
    
    # train model
    trainer = L.Trainer(precision="16-mixed", 
                        callbacks=callbacks, 
                        logger=logger,
                        max_epochs=args.epochs, 
                        accumulate_grad_batches=args.accumulate_grad_batches,
                        devices=args.gpus,
                        num_nodes=args.nodes,
                        accelerator="gpu",
                        strategy="ddp")
    
    if resume:
        # resume training from checkpoint
        cp_files = glob.glob(f'{args.model_path}/{args.model_id}*')
        if not cp_files:
            raise FileNotFoundError("No checkpoint found")

        cp_path = cp_files[0]
        print(f"Found checkpoint: {cp_path}")

        trainer.fit(model=model, train_dataloaders=train_data, val_dataloaders=val_data, ckpt_path=cp_path)
    
    else:
        # train new model 
        trainer.fit(model=model, train_dataloaders=train_data, val_dataloaders=val_data)


# model evaluation
def evaluate(args, data):
    val_shape = data["val_shape_out"]

    model = Model(args, val_shape)

    # load model
    cp_files = glob.glob(f'{args.model_path}{args.model_id}*')
    if not cp_files:
        raise FileNotFoundError("No checkpoint found")

    cp_path = cp_files[0]
    print(f"Found checkpoint: {cp_path}")
    checkpoint = torch.load(cp_path)

    model.load_state_dict(checkpoint["state_dict"])
    model = model.to('cuda')
    model.eval()

    evaluate_model(data, args, model.model)

