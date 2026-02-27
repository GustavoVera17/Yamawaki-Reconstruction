import os
import glob
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Importamos los módulos
from dataset_cassi import CASSIDataset
from yamawaki_net import YamawakiNet
from metricas import calcular_psnr, calcular_sam
from skimage.metrics import structural_similarity as ssim_metric

def calcular_ssim(pred, target):
    pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0) 
    target_np = target.detach().cpu().numpy().transpose(1, 2, 0)
    ssim_val = ssim_metric(target_np, pred_np, data_range=1.0, channel_axis=-1)
    return float(ssim_val)

def cargar_ultimo_checkpoint(checkpoint_dir, modelo, device):
    """Busca y carga el archivo .pth con la época más alta."""
    archivos_pth = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not archivos_pth:
        raise FileNotFoundError("No se encontraron modelos .pth en la carpeta checkpoints.")
    
    # Ordenar por número de época (extraído del nombre del archivo)
    ultimo_pth = max(archivos_pth, key=os.path.getctime)
    print(f"[*] Cargando pesos del modelo: {os.path.basename(ultimo_pth)}")
    
    modelo.load_state_dict(torch.load(ultimo_pth, map_location=device))
    modelo.eval()
    return modelo

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando inferencia en: {device}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta_test = os.path.join(BASE_DIR, "dataset", "fortest")
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")

    # 1. Cargar Dataset (Solo Test)
    print("Cargando imágenes de prueba...")
    dataset_test = CASSIDataset(root_dir=ruta_test, patch_size=48, num_patches_per_img=1, is_train=False)

    # 2. Inicializar y cargar el Modelo
    modelo = YamawakiNet(in_channels=1, out_channels=31, num_features=64, num_blocks=9, S=2).to(device)
    modelo = cargar_ultimo_checkpoint(checkpoint_dir, modelo, device)

    # 3. Extraer una imagen al azar y hacer la predicción
    idx_azar = random.randint(0, len(dataset_test.image_folders) - 1)
    nombre_carpeta = os.path.basename(dataset_test.image_folders[idx_azar])
    print(f"Evaluando imagen: {nombre_carpeta}")

    with torch.no_grad():
        in_2d, gt_3d = dataset_test.get_full_image(idx_azar)
        in_2d, gt_3d = in_2d.to(device), gt_3d.to(device)
        
        pred_3d = modelo(in_2d)
        
        # Calcular métricas del cubo completo
        psnr_val = calcular_psnr(pred_3d, gt_3d)
        ssim_val = calcular_ssim(pred_3d[0], gt_3d[0])
        sam_val = calcular_sam(pred_3d, gt_3d)

    # 4. Preparar tensores para Matplotlib (Pasar a CPU y NumPy)
    img_pan = in_2d[0, 0].cpu().numpy()
    cubo_gt = gt_3d[0].cpu().numpy()     # Dimensiones: (31, H, W)
    cubo_pred = pred_3d[0].cpu().numpy() # Dimensiones: (31, H, W)
    
    # 5. Configurar el Dashboard Interactivo
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.canvas.manager.set_window_title(f'Inferencia CASSI - {nombre_carpeta}')
    plt.subplots_adjust(bottom=0.25, top=0.85) # Espacio para el slider abajo y título arriba

    # Título principal con métricas globales
    fig.suptitle(f"Imagen: {nombre_carpeta} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | SAM: {sam_val:.2f}°", 
                 fontsize=14, fontweight='bold')

    # Longitudes de onda aproximadas para CAVE (400nm a 700nm en pasos de 10nm)
    longitudes_onda = np.arange(400, 710, 10)
    banda_inicial = 15 # Empezamos a la mitad (550nm - Verde)

    # Dibujar paneles iniciales
    axes[0].set_title("Medición CASSI PAN")
    axes[0].imshow(img_pan, cmap='gray')
    axes[0].axis('off')

    titulo_gt = axes[1].set_title(f"Verdad Original ({longitudes_onda[banda_inicial]}nm)")
    img_gt_plot = axes[1].imshow(cubo_gt[banda_inicial], cmap='gray', vmin=0, vmax=1)
    axes[1].axis('off')

    titulo_pred = axes[2].set_title(f"Predicción DFHM ({longitudes_onda[banda_inicial]}nm)")
    img_pred_plot = axes[2].imshow(cubo_pred[banda_inicial], cmap='gray', vmin=0, vmax=1)
    axes[2].axis('off')

    error_inicial = np.abs(cubo_gt[banda_inicial] - cubo_pred[banda_inicial])
    axes[3].set_title("Mapa de Error Absoluto")
    img_error_plot = axes[3].imshow(error_inicial, cmap='inferno', vmin=0, vmax=0.2) # vmax bajo para resaltar errores
    fig.colorbar(img_error_plot, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].axis('off')

    # 6. Crear el Slider
    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor=axcolor)
    slider_banda = Slider(
        ax=ax_slider,
        label='Banda Espectral',
        valmin=0,
        valmax=30,
        valinit=banda_inicial,
        valstep=1
    )

    # 7. Función de actualización del slider
    def update(val):
        b = int(slider_banda.val)
        wl = longitudes_onda[b]
        
        # Actualizar los datos de las imágenes
        img_gt_plot.set_data(cubo_gt[b])
        img_pred_plot.set_data(cubo_pred[b])
        img_error_plot.set_data(np.abs(cubo_gt[b] - cubo_pred[b]))
        
        # Actualizar títulos
        titulo_gt.set_text(f"Verdad Original ({wl}nm)")
        titulo_pred.set_text(f"Predicción DFHM ({wl}nm)")
        
        fig.canvas.draw_idle()

    slider_banda.on_changed(update)
    plt.show()

if __name__ == "__main__":
    main()