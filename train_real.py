import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib
# 🛑 ACTIVAMOS MODO HEADLESS: Genera gráficos en memoria, sin ventanas emergentes
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric
from tqdm import tqdm

from dataset_cassi import CASSIDataset
from yamawaki_net_real import YamawakiNet # Asegúrate de que este archivo contenga la red fiel
from metricas import calcular_psnr, calcular_sam

def calcular_ssim(pred, target):
    pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0) 
    target_np = target.detach().cpu().numpy().transpose(1, 2, 0)
    ssim_val = ssim_metric(target_np, pred_np, data_range=1.0, channel_axis=-1)
    return float(ssim_val) 

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta_train = os.path.join(BASE_DIR, "dataset", "fortrain")
    ruta_test = os.path.join(BASE_DIR, "dataset", "fortest")
    
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Configuraciones para la red gigante
    BATCH_SIZE = 4 
    MAX_EPOCHS = 5000     
    INITIAL_LR = 1e-4
    
    # ⚙️ PARÁMETROS DE ENTRENAMIENTO MODERNO
    PACIENCIA_EARLY_STOP = 50 # Épocas sin mejorar antes de abortar
    PACIENCIA_LR = 15         # Épocas sin mejorar antes de reducir el Learning Rate

    print("Cargando datasets a la RAM...")
    dataset_train = CASSIDataset(root_dir=ruta_train, patch_size=48, num_patches_per_img=50)
    dataset_test = CASSIDataset(root_dir=ruta_test, patch_size=48, num_patches_per_img=1) 
    
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    # Instanciamos la red FIEL (512 canales)
    modelo = YamawakiNet(in_channels=1, out_channels=31, num_features=512, num_blocks=9, S=2).to(device)
    
    criterio = nn.L1Loss() 
    
    # Añadimos Weight Decay (L2) para evitar que memorice el ruido
    optimizador = optim.Adam(modelo.parameters(), lr=INITIAL_LR, weight_decay=1e-5)
    
    # Scheduler: Reduce el LR si el loss de validación se estanca
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='min', factor=0.5, patience=PACIENCIA_LR, verbose=True)

    hist_loss, hist_psnr, hist_ssim, hist_sam = [], [], [], []

    print("\n¡Iniciando el entrenamiento de la arquitectura FIEL (Headless Mode)!")
    
    idx_azar = random.randint(0, len(dataset_test.image_folders) - 1)
    nombre_imagen = os.path.basename(dataset_test.image_folders[idx_azar])
    print(f"[*] Imagen fija seleccionada para validación y Dashboard: {nombre_imagen}\n")
    
    mejor_loss_val = float('inf')
    epocas_sin_mejora = 0

    for epoch in range(MAX_EPOCHS):
        modelo.train()
        loss_epoch = 0.0
        
        loop_entrenamiento = tqdm(loader_train, desc=f"Época [{epoch+1}/{MAX_EPOCHS}]", leave=True)
        
        for batch_idx, (entrada_2d, objetivo_3d) in enumerate(loop_entrenamiento):
            entrada_2d, objetivo_3d = entrada_2d.to(device), objetivo_3d.to(device)
            
            optimizador.zero_grad()
            prediccion_3d = modelo(entrada_2d)
            
            loss = criterio(prediccion_3d, objetivo_3d)
            loss.backward()
            optimizador.step()
            
            loss_epoch += loss.item()
            loop_entrenamiento.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = float(loss_epoch / len(loader_train))
        hist_loss.append(avg_loss)

        # -- VALIDACIÓN Y MÉTRICAS --
        modelo.eval()
        with torch.no_grad(): 
            test_in_2d, test_gt_3d = dataset_test.get_full_image(idx_azar)
            test_in_2d, test_gt_3d = test_in_2d.to(device), test_gt_3d.to(device)
            
            test_pred_3d = modelo(test_in_2d)
            
            # Usamos el L1 Loss de la imagen completa como métrica de control real
            val_loss = criterio(test_pred_3d, test_gt_3d).item()
            
            val_psnr = float(calcular_psnr(test_pred_3d, test_gt_3d))
            val_sam = float(calcular_sam(test_pred_3d, test_gt_3d))
            val_ssim = float(calcular_ssim(test_pred_3d[0], test_gt_3d[0]))
            
            hist_psnr.append(val_psnr)
            hist_sam.append(val_sam)
            hist_ssim.append(val_ssim)

        # Pasamos el loss de validación al Scheduler
        scheduler.step(val_loss)

        # 🚨 LÓGICA DE EARLY STOPPING Y GUARDADO DEL MEJOR MODELO
        if val_loss < mejor_loss_val:
            mejor_loss_val = val_loss
            epocas_sin_mejora = 0
            # Solo guardamos el modelo si realmente superó su récord
            torch.save(modelo.state_dict(), os.path.join(checkpoint_dir, "yamawaki_fiel_best.pth"))
            print(f"⭐ ¡Nuevo récord de validación! Modelo guardado. (Loss: {val_loss:.5f})")
        else:
            epocas_sin_mejora += 1
            print(f"⚠️ Sin mejora por {epocas_sin_mejora} épocas. (Mejor: {mejor_loss_val:.5f})")

        print(f"Resumen Época {epoch+1} | Loss Train: {avg_loss:.4f} | PSNR: {val_psnr:.2f}dB | SSIM: {val_ssim:.4f} | SAM: {val_sam:.2f}°\n")

        # =====================================================================
        # EXPORTACIÓN DEL DASHBOARD PNG (CADA 10 ÉPOCAS)
        # =====================================================================
        if (epoch + 1) % 10 == 0:
            fig = plt.figure(figsize=(16, 8))
            
            img_cassi = test_in_2d[0, 0].cpu().numpy()
            banda_idx = 20
            img_gt = test_gt_3d[0, banda_idx].cpu().numpy()
            img_pred = test_pred_3d[0, banda_idx].cpu().numpy()
            img_error = np.abs(img_gt - img_pred)

            ax1 = fig.add_subplot(2, 4, 1)
            ax1.set_title("Medición CASSI PAN")
            ax1.imshow(img_cassi, cmap='gray')
            ax1.axis('off')

            ax2 = fig.add_subplot(2, 4, 2)
            ax2.set_title(f"Verdad Limpia (Banda {banda_idx})")
            ax2.imshow(img_gt, cmap='gray', vmin=0, vmax=1) 
            ax2.axis('off')

            ax3 = fig.add_subplot(2, 4, 3)
            ax3.set_title(f"Predicción (Banda {banda_idx})")
            ax3.imshow(img_pred, cmap='gray', vmin=0, vmax=1) 
            ax3.axis('off')

            ax4 = fig.add_subplot(2, 4, 4)
            ax4.set_title("Error (Prueba - Predicción)")
            im = ax4.imshow(img_error, cmap='inferno') 
            fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
            ax4.axis('off')

            ax5 = fig.add_subplot(2, 4, 5)
            ax5.set_title("L1 Loss (Train)")
            ax5.plot(hist_loss, color='red')
            ax5.grid(True)

            ax6 = fig.add_subplot(2, 4, 6)
            ax6.set_title("PSNR (Validación)")
            ax6.plot(hist_psnr, color='blue')
            ax6.grid(True)

            ax7 = fig.add_subplot(2, 4, 7)
            ax7.set_title("SSIM (Validación)")
            ax7.plot(hist_ssim, color='green')
            ax7.grid(True)

            ax8 = fig.add_subplot(2, 4, 8)
            ax8.set_title("SAM (Validación)")
            ax8.plot(hist_sam, color='purple')
            ax8.grid(True)

            fig.suptitle(f"Reporte Automático: {nombre_imagen} | Época: {epoch+1}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Guardamos la figura en el disco y cerramos la memoria
            ruta_dashboard = os.path.join(BASE_DIR, "dashboard.png")
            plt.savefig(ruta_dashboard, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"📸 Dashboard exportado exitosamente en: {ruta_dashboard}")

        # Ejecutador del límite de Early Stopping
        if epocas_sin_mejora >= PACIENCIA_EARLY_STOP:
            print(f"\n🛑 ENTRENAMIENTO DETENIDO POR EARLY STOPPING.")
            print(f"La red dejó de aprender durante {PACIENCIA_EARLY_STOP} épocas consecutivas.")
            break

if __name__ == "__main__":
    train()