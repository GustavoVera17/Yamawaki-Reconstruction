import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CASSIDataset(Dataset):
    def __init__(self, root_dir, patch_size=48, num_patches_per_img=100, is_train=True):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.num_patches = num_patches_per_img
        self.is_train = is_train
        
        self.image_folders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
        print(f"Encontradas {len(self.image_folders)} carpetas de imágenes en {root_dir}")
        
        # Máscara de Bernoulli (p=0.5) estática
        self.mask = np.random.binomial(1, 0.5, (512, 512)).astype(np.float32)

    def _load_cave_cube(self, folder_path):
        """Carga las bandas PNG buscando inteligentemente en cualquier subcarpeta"""
        # El comodín '**' le dice a Python que busque en todos los subdirectorios
        search_pattern = os.path.join(folder_path, '**', '*.png')
        all_pngs = sorted(glob.glob(search_pattern, recursive=True))
        
        band_files = []
        for f in all_pngs:
            # Extraemos solo el nombre del archivo (ej. 'apples_ms_01.png')
            filename = os.path.basename(f)
            name_no_ext = os.path.splitext(filename)[0]
            
            # Las bandas de CAVE terminan en números (01 a 31). Ignoramos los '_RGB'
            if name_no_ext[-2:].isdigit():
                band_files.append(f)
                
        # Aseguramos el orden y tomamos exactamente las 31 bandas
        band_files = sorted(band_files)[:31]
        
        # Seguro contra errores de carpetas incompletas
        if len(band_files) != 31:
            raise ValueError(f"Error en {folder_path}: Se encontraron {len(band_files)} bandas numéricas, pero se necesitan 31.")
        
        cube = []
        for file in band_files:
            img = Image.open(file).convert('L')
            img_array = np.array(img, dtype=np.float32) / 255.0 
            cube.append(img_array)
            
        return np.stack(cube, axis=2) # (512, 512, 31)

    def _simulate_cassi(self, cube):
        h, w, bands = cube.shape
        masked_cube = cube * self.mask[:, :, np.newaxis]
        
        # Simulación del prisma (Ensanchamiento de la imagen)
        shifted_cube = np.zeros((h, w + (bands - 1), bands), dtype=np.float32)
        for i in range(bands):
            shifted_cube[:, i:(i + w), i] = masked_cube[:, :, i]
            
        measurement = np.sum(shifted_cube, axis=2)
        return measurement, shifted_cube

    def __len__(self):
        return len(self.image_folders) * self.num_patches

    def __getitem__(self, idx):
        img_idx = idx // self.num_patches
        folder_path = self.image_folders[img_idx]
        
        cube = self._load_cave_cube(folder_path)
        measurement, shifted_cube = self._simulate_cassi(cube)
        
        max_h = measurement.shape[0] - self.patch_size
        max_w = measurement.shape[1] - self.patch_size
        
        rng = np.random.RandomState(idx) 
        start_h = rng.randint(0, max_h)
        start_w = rng.randint(0, max_w)
        
        patch_meas = measurement[start_h : start_h + self.patch_size, 
                                 start_w : start_w + self.patch_size]
        patch_cube = shifted_cube[start_h : start_h + self.patch_size, 
                                  start_w : start_w + self.patch_size, :]
        
        tensor_meas = torch.from_numpy(patch_meas).unsqueeze(0) # (1, 48, 48)
        tensor_cube = torch.from_numpy(patch_cube).permute(2, 0, 1) # (31, 48, 48)
        
        return tensor_meas, tensor_cube

# ==========================================
# ZONA DE PRUEBA
# ==========================================
if __name__ == "__main__":
    import os
    
    # 1. Obtenemos la ruta exacta de donde está guardado este script (Yamawaki_Rep)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Construimos la ruta absoluta hacia tus datos
    ruta_train = os.path.join(BASE_DIR, "dataset", "fortrain")
    
    print(f"Buscando datos en: {ruta_train}")
    print("Iniciando prueba del Dataset CASSI...")
    
    try:
        dataset_train = CASSIDataset(root_dir=ruta_train, patch_size=48, num_patches_per_img=10)
        entrada_2d, salida_3d = dataset_train[0]
        
        print(f"\n¡Éxito! Se ha generado un parche de entrenamiento:")
        print(f"Input (Medición CASSI 2D): {entrada_2d.shape}")
        print(f"Ground Truth (Cubo HSI)  : {salida_3d.shape}")
        
    except Exception as e:
        print(f"\nHubo un error: {e}")