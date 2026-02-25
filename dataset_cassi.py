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
        
        self.mask = np.random.binomial(1, 0.5, (512, 512)).astype(np.float32)

        self.cubes_cache = []
        self.meas_cache = []
        
        print(f"Pre-cargando {len(self.image_folders)} imágenes a la RAM. Por favor espera...")
        for folder in self.image_folders:
            cube = self._load_cave_cube(folder)
            measurement, shifted_clean = self._simulate_cassi(cube)
            self.meas_cache.append(measurement)
            self.cubes_cache.append(shifted_clean)
        print("¡Precarga completada!")

    def _load_cave_cube(self, folder_path):
        search_pattern = os.path.join(folder_path, '**', '*.png')
        all_pngs = sorted(glob.glob(search_pattern, recursive=True))
        
        band_files = []
        for f in all_pngs:
            filename = os.path.basename(f)
            name_no_ext = os.path.splitext(filename)[0]
            if name_no_ext[-2:].isdigit():
                band_files.append(f)
                
        band_files = sorted(band_files)[:31]
        
        if len(band_files) != 31:
            raise ValueError(f"Error en {folder_path}: Se encontraron {len(band_files)} bandas, se necesitan 31.")
        
        cube = []
        for file in band_files:
            img = Image.open(file)
            img_array = np.array(img, dtype=np.float32)
            
            # --- NUEVO: Seguro contra "RGB falso" ---
            # Si la imagen tiene 3 dimensiones (H, W, 3), tomamos solo el primer canal (H, W)
            if img_array.ndim == 3:
                img_array = img_array[:, :, 0]
                
            # Normalización Inteligente (Preservando 16-bits si existen)
            max_val = 65535.0 if np.max(img_array) > 255.0 else 255.0
            img_array = img_array / max_val
            
            cube.append(img_array)
            
        return np.stack(cube, axis=2)
    
    def _simulate_cassi(self, cube):
        h, w, bands = cube.shape
        masked_cube = cube * self.mask[:, :, np.newaxis]
        
        shifted_masked = np.zeros((h, w + (bands - 1), bands), dtype=np.float32)
        shifted_clean = np.zeros((h, w + (bands - 1), bands), dtype=np.float32)
        
        for i in range(bands):
            shifted_masked[:, i:(i + w), i] = masked_cube[:, :, i]
            shifted_clean[:, i:(i + w), i] = cube[:, :, i] 
            
        measurement = np.sum(shifted_masked, axis=2)
        return measurement, shifted_clean

    def __len__(self):
        return len(self.image_folders) * self.num_patches

    def __getitem__(self, idx):
        img_idx = idx // self.num_patches
        measurement = self.meas_cache[img_idx]
        shifted_cube = self.cubes_cache[img_idx]
        
        max_h = measurement.shape[0] - self.patch_size
        max_w = measurement.shape[1] - self.patch_size
        
        rng = np.random.RandomState(idx) 
        start_h = rng.randint(0, max_h)
        start_w = rng.randint(0, max_w)
        
        patch_meas = measurement[start_h : start_h + self.patch_size, start_w : start_w + self.patch_size]
        patch_cube = shifted_cube[start_h : start_h + self.patch_size, start_w : start_w + self.patch_size, :]
        
        tensor_meas = torch.from_numpy(patch_meas).unsqueeze(0) 
        tensor_cube = torch.from_numpy(patch_cube).permute(2, 0, 1) 
        return tensor_meas, tensor_cube

    def get_full_image(self, img_idx):
        measurement = self.meas_cache[img_idx]
        shifted_cube = self.cubes_cache[img_idx]
        tensor_meas = torch.from_numpy(measurement).unsqueeze(0).unsqueeze(0)
        tensor_cube = torch.from_numpy(shifted_cube).permute(2, 0, 1).unsqueeze(0)
        return tensor_meas, tensor_cube