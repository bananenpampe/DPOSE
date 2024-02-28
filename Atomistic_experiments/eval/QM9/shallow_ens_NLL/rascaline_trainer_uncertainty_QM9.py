import torch
import pytorch_lightning as pl

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "model"))

from metatensor.torch import TensorBlock, Labels, TensorMap
from nn.model import BPNNModel
from nn.interaction import BPNNInteraction
from nn.aggregation import BPNNStructureWiseAggregationVar
from nn.response import ForceUncertaintyRespone, UnitResponse
from nn.loss import EnergyForceLoss, GeneralLossUQ, EnergyForceUncertaintyLoss, CRPS
from transformer.composition import CompositionTransformer


class BPNNRascalineModule(pl.LightningModule):
    
    def __init__(self,
                 example_tensormap,
                 energy_transformer=CompositionTransformer(),
                 model = BPNNModel(
                 interaction=BPNNInteraction(n_out=64, 
                                             n_hidden= 256,
                                             n_hidden_layers = 3,
                                             activation = torch.nn.SiLU,),
                 aggregation=BPNNStructureWiseAggregationVar(),
                 response=UnitResponse()
                 ),
                 loss_fn = GeneralLossUQ,
                 regularization=1e-03,
                 w_force_uncertainty=False):
        
        super().__init__()
        #print(regularization)
        #self.save_hyperparameters({'l2 reg': regularization})
        self.model = model
        self.model.initialize_weights(example_tensormap)
        
        self.loss_fn = loss_fn(base_loss=CRPS)

        self.loss_rmse = EnergyForceLoss(w_forces=False,
                                base_loss=torch.nn.MSELoss)
        
        self.loss_mae = EnergyForceLoss(w_forces=False,
                                base_loss=torch.nn.L1Loss)
        
        self.energy_transformer = energy_transformer
        self.regularization = regularization

    def forward(self, feats, systems):
        return self.model(feats, systems)
    
    def calculate(self, feats, systems):
        
        outputs = self(feats, systems)

        outputs = self.energy_transformer.inverse_transform(systems, outputs)

        return outputs
    

    def training_step(self, batch, batch_idx):
        feats, properties, systems = batch

        batch_size = len(systems)

        #print(feats.block(0).values.requires_grad)

        properties = self.energy_transformer.transform(systems, properties)

        outputs = self(feats,systems)
        loss = self.loss_fn(outputs, properties)

        self.log('train_loss', loss, enable_graph=True, batch_size=batch_size)
        
        """
        for param in self.parameters():
            loss += self.regularization * torch.norm(param)**2
        """

        return loss

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)
        
    def on_test_model_eval(self, *args, **kwargs):
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def validation_step(self, batch, batch_idx):

        feats, properties, systems = batch 

        batch_size = len(systems)

        outputs = self(feats, systems)
        outputs = TensorMap(properties.keys,[outputs.block(0).copy()])

        outputs = self.energy_transformer.inverse_transform(systems, outputs)

        energy_val_mse, forces_val_mse = self.loss_rmse.report(outputs, properties)
        energy_val_mae, forces_val_mae = self.loss_mae.report(outputs, properties)

        loss = energy_val_mse + forces_val_mse

        self.log('val_loss',
                 loss.item(),
                 batch_size=batch_size,
                 on_epoch=True)
        
        self.log("val_energy_mae",
                 torch.clone(energy_val_mae),
                 batch_size=batch_size,
                 on_epoch=True)
        
        
        self.log("val_energy_mse",
                 torch.clone(energy_val_mse),
                 batch_size=batch_size,
                 on_epoch=True)

        # log rmse
        self.log("val_energy_rmse",
                 torch.sqrt(torch.clone(energy_val_mse)),
                 batch_size=batch_size,
                 on_epoch=True)


        return loss

    def test_step(self, batch, batch_idx):
        
        feats, properties, systems = batch


        batch_size = len(systems)

        outputs = self(feats, systems)
        #for now:

        outputs = TensorMap(properties.keys,[outputs.block(0)])

        outputs = self.energy_transformer.inverse_transform(systems, outputs)

        energy_test_mse, forces_test_mse = self.loss_rmse.report(outputs, properties)
        energy_test_mae, forces_test_mae = self.loss_mae.report(outputs, properties)

        loss = energy_test_mse + forces_test_mse

        self.log('test_loss',
                loss.item(),
                batch_size=batch_size,
                on_epoch=True)

        self.log("test_energy_mae",
                torch.clone(energy_test_mae),
                batch_size=batch_size,
                on_epoch=True)
        
        self.log("test_energy_mse",
                torch.clone(energy_test_mse),
                batch_size=batch_size,
                on_epoch=True)

        # log rmse
        self.log("test_energy_rmse",
                torch.sqrt(torch.clone(energy_test_mse)),
                batch_size=batch_size,
                on_epoch=True)
        

        return loss

    def backward(self, loss, *args, **kwargs):
        loss.backward(retain_graph=True)

    """ 
    def configure_optimizers(self):
        optimizer = torch.optim.LBFGS(self.parameters(), lr=1., line_search_fn="strong_wolfe")
        return optimizer
    """

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-3, amsgrad=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                        patience=20,
                                                                        factor=0.5,
                                                                        verbose=True,
                                                                        min_lr=1e-05),
                "monitor": "val_energy_mae",
                "interval": "epoch",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
