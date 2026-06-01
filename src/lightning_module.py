import torch
import torch.nn.functional as F
import lightning as L
import torchmetrics

class SimpsonsModule(L.LightningModule):
    """Lightning wrapper que orquesta el entrenamiento de la CNN"""
    
    def __init__(self, cfg, model: torch.nn.Module, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.cfg = cfg
        self.model = model
        self.class_weights = class_weights
        
        # Metricas 

        # Usar avg weighted y no macro porque en train hay clases que no hay en test
        # Si predice clases ausentes tendrán soporte=0, anulando su penalizacion matemática en el promedio
        metrics_kws = {"task": "multiclass", "num_classes": cfg.num_classes, "average": "weighted"}
        
        # training
        self.train_acc = torchmetrics.Accuracy(**metrics_kws)
        
        # val
        self.val_acc = torchmetrics.Accuracy(**metrics_kws)
        self.val_f1 = torchmetrics.F1Score(**metrics_kws)
        
        # test
        self.test_acc = torchmetrics.Accuracy(**metrics_kws)
        self.test_f1 = torchmetrics.F1Score(**metrics_kws)
        self.test_precision = torchmetrics.Precision(**metrics_kws)
        self.test_recall = torchmetrics.Recall(**metrics_kws)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # Asignar pesos de clase
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.device)
        
        # Funcion de perdida
        loss = F.cross_entropy(logits, y, weight=self.class_weights, label_smoothing=0.1)
        preds = torch.argmax(logits, dim=1)

        # Sacar metricas
        self.train_acc(preds, y)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # Asignar pesos de clase
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.device)
            
        # Funcion de perdida
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        preds = torch.argmax(logits, dim=1)

        # Sacar metricas
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("val/f1_weighted", self.val_f1, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # Asignar pesos de clase
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.device)

        # Funcion de perdida
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Sacar metricas
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)

        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc, on_epoch=True)
        self.log("test/f1_weighted", self.test_f1, on_epoch=True)
        self.log("test/precision", self.test_precision, on_epoch=True)
        self.log("test/recall", self.test_recall, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Optimizador
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )
        
        # OneCycleLR ajusta de forma dinamica el learning rate
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.lr,
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
