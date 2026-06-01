import os
import tarfile
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L
import torchvision.datasets as datasets
from torchvision.transforms import v2

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
        
        # Saltar carpetas vacias
        valid_classes = []
        for c in classes:
            class_dir = os.path.join(directory, c)
            if any(os.scandir(class_dir)):
                valid_classes.append(c)
                
        class_to_idx = {cls_name: i for i, cls_name in enumerate(valid_classes)}
        return valid_classes, class_to_idx

class SimpsonsTestDataset(Dataset):
    """Clase de personalizada para el set de prueba que no tiene subcarpetas"""
    def __init__(self, directory, class_to_idx, transform=None):
        self.directory = Path(directory)
        self.class_to_idx = class_to_idx
        self.transform = transform
        
        self.samples = []
        for file_path in self.directory.glob("*.jpg"):
            # extraer la clase del nombre del archivo (ej: abraham_grampa_simpson_0.jpg)
            class_name = "_".join(file_path.stem.split("_")[:-1])
            if class_name in self.class_to_idx:
                self.samples.append((str(file_path), self.class_to_idx[class_name]))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class SimpsonsDataModule(L.LightningDataModule):
    def __init__(self, cfg, train_transforms=None, val_transforms=None):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)
        # directorio train
        self.train_dir = self.data_dir / "simpsons"
        # directorio test
        self.test_dir = self.data_dir / "simpsons_testset"
        
        size = self.cfg.image_size
        
        # Si el notebook inyecta transformaciones, las usamos
        self.train_transforms = train_transforms if train_transforms is not None else v2.Compose([
            v2.Resize((size, size)),
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
        self.num_classes = cfg.num_classes

    def prepare_data(self):
        """Descomprime los archivos"""
        for archive_name, extract_path in [("simpsons_train.tar.gz", self.train_dir), ("simpsons_test.tar.gz", self.test_dir)]:
            archive_path = self.data_dir / archive_name
            if archive_path.exists() and not extract_path.exists():
                print(f"Extrayendo {archive_name}")
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=self.data_dir)
                print(f"{archive_name} extraido en {self.data_dir}")

    def setup(self, stage=None):
        # CustomImageFolder ignora carpetas vacias
        full_train_ds = CustomImageFolder(self.train_dir)
        self.classes = full_train_ds.classes
        self.num_classes = len(self.classes)
        # Actualizamos config
        self.cfg.num_classes = self.num_classes 

        # Split train/val
        train_size = int(0.8 * len(full_train_ds))
        val_size = len(full_train_ds) - train_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            full_train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        
        train_samples = [full_train_ds.samples[i] for i in train_subset.indices]
        val_samples = [full_train_ds.samples[i] for i in val_subset.indices]
                
        self.train_ds = SimpsonsDataset(train_samples, transform=self.train_transforms)
        self.val_ds = SimpsonsDataset(val_samples, transform=self.val_test_transforms)
        self.test_ds = SimpsonsTestDataset(self.test_dir, full_train_ds.class_to_idx, transform=self.val_test_transforms)

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
