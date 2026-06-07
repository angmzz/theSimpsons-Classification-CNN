import os
import tarfile
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L
import torchvision.datasets as datasets
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

class SimpsonsDataset(Dataset):
    """Wrapper para aplicar transformaciones sobre una lista de samples usando PIL y torchvision."""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class CustomImageFolder(datasets.ImageFolder):
    """Clase personalizada de ImageFolder que ignora las carpetas sin imagenes"""
    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"No se encontraron carpetas de clases en {directory}")
        
        # Saltar carpetas vacias o con muy pocas imágenes (menos de 50)
        min_samples = 50
        valid_classes = []
        for c in classes:
            class_dir = os.path.join(directory, c)
            # Contar archivos
            num_images = sum(1 for entry in os.scandir(class_dir) if entry.is_file())
            if num_images >= min_samples:
                valid_classes.append(c)
                
        class_to_idx = {cls_name: i for i, cls_name in enumerate(valid_classes)}
        return valid_classes, class_to_idx


class SimpsonsDataModule(L.LightningDataModule):
    def __init__(self, cfg, train_transforms=None, val_transforms=None):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)
        # Directorio único de base de datos
        self.train_dir = self.data_dir / "simpsons"
        
        size = self.cfg.image_size
        
        # Si el notebook inyecta transformaciones, las usamos
        self.train_transforms = train_transforms if train_transforms is not None else v2.Compose([
            v2.Resize((size, size)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        self.val_test_transforms = val_transforms if val_transforms is not None else v2.Compose([
            v2.Resize((size, size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.classes = []
        self.num_classes = 0 # Se calculará dinámicamente en setup()

    def prepare_data(self):
        """Descomprime el archivo """
        archive_name, extract_path = "simpsons_train.tar.gz", self.train_dir
        archive_path = self.data_dir / archive_name
        if archive_path.exists() and not extract_path.exists():
            print(f"Extrayendo {archive_name}")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=self.data_dir)
            print(f"{archive_name} extraido en {self.data_dir}")

    def setup(self, stage=None):
        # Cargamos toda la base de datos (se ignoran carpetas vacías)
        full_ds = CustomImageFolder(self.train_dir)
        self.classes = full_ds.classes
        self.num_classes = len(self.classes)
        print(f"\n[INFO] Escaneo completado: Se encontraron {self.num_classes} clases válidas (con 50+ imágenes).")

        # Extraemos targets (y) para realizar un split estratificado
        samples = full_ds.samples
        targets = [label for _, label in samples]

        # 1. Separamos 80% para (Train+Val) y 20% para Test intacto
        train_val_samples, test_samples, train_val_targets, _ = train_test_split(
            samples, targets, test_size=0.20, random_state=42, stratify=targets
        )

        # 2. Del 80% restante, sacamos un 12.5% (que equivale al 10% del total) para Validación
        train_samples, val_samples, _, _ = train_test_split(
            train_val_samples, train_val_targets, test_size=0.125, random_state=42, stratify=train_val_targets
        )
                
        self.train_ds = SimpsonsDataset(train_samples, transform=self.train_transforms)
        self.val_ds = SimpsonsDataset(val_samples, transform=self.val_test_transforms)
        self.test_ds = SimpsonsDataset(test_samples, transform=self.val_test_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.cfg.batch_size, shuffle=True, 
            num_workers=4, pin_memory=True, persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.cfg.batch_size, shuffle=False, 
            num_workers=4, pin_memory=True, persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.cfg.batch_size, shuffle=False, 
            num_workers=4, pin_memory=True, persistent_workers=True
        )
