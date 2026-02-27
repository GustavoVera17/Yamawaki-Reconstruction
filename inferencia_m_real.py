import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Importamos tus módulos
from dataset_cassi import CASSIDataset
from yamawaki_net_real import YamawakiNet # <--- Importamos el archivo REAL
from metricas import calcular_psnr, calcular_sam
from skimage.metrics import structural_similarity as ssim_metric

def calcular_ssim(pred, target):
    pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0) 
    target_np = target.detach().cpu().numpy().transpose(1, 2, 0)
    ssim_val = ssim_metric(target_np, pred_np, data_range=1.0, channel_axis=-1)
    return float(ssim_val)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando inferencia manual en: {device}")

    # =========================================================================
    # ⚙️ CONFIGURACIÓN MANUAL (Modifica estas dos rutas según lo que necesites)
    # =========================================================================
    
    # 1. Ruta exacta del modelo .pth que quieres usar (¡Debe ser uno entrenado con la red de 512 canales!)
    RUTA_MODELO_PTH = r"C:\CODE2026\CASSIproy2026\CASSIproy2026\Yamawaki_Rep\checkpoints\yamawaki_real_epoch_100.pth"
    
    # 2. Ruta exacta de la carpeta de la imagen que quieres evaluar
    RUTA_CARPETA_IMAGEN = r"C:\CODE2026\CASSIproy2026\CASSIproy2026\Yamawaki_Rep\dataset\fortest\oil_painting_ms"

    # =========================================================================

    if not os.path.exists(RUTA_MODELO_PTH):
        raise FileNotFoundError(f"No se encontró el modelo en: {RUTA_MODELO_PTH}")
    if not os.path.exists(RUTA_CARPETA_IMAGEN):
        raise FileNotFoundError(f"No se encontró la carpeta de imagen en: {RUTA_CARPETA_IMAGEN}")

    # Extraemos la ruta padre y el nombre de la carpeta de la imagen
    ruta_padre = os.path.dirname(RUTA_CARPETA_IMAGEN)
    nombre_imagen_objetivo = os.path.basename(RUTA_CARPETA_IMAGEN)

    # 1. Cargar Dataset
    print(f"Cargando el entorno de validación desde: {ruta_padre}...")
    dataset_test = CASSIDataset(root_dir=ruta_padre, patch_size=48, num_patches_per_img=1, is_train=False)

    # Buscamos el índice exacto
    idx_elegido = None
    for i, folder in enumerate(dataset_test.image_folders):
        if os.path.basename(folder) == nombre_imagen_objetivo:
            idx_elegido = i
            break
            
    if idx_elegido is None:
        raise ValueError(f"No se pudo encontrar '{nombre_imagen_objetivo}' dentro del dataset cargado.")

    print(f"[*] Evaluando imagen: {nombre_imagen_objetivo}")
    print(f"[*] Usando pesos: {os.path.basename(RUTA_MODELO_PTH)}")

    # 2. Inicializar y cargar el Modelo REAL
    # <--- AUMENTAMOS A 512 CANALES AQUÍ --->
    modelo = YamawakiNet(in_channels=1, out_channels=31, num_features=512, num_blocks=9, S=2).to(device)
    
    # Cargamos los pesos
    modelo.load_state_dict(torch.load(RUTA_MODELO_PTH, map_location=device))
    modelo.eval()

    # 3. Extraer la imagen específica y hacer la predicción
    with torch.no_grad():
        in_2d, gt_3d = dataset_test.get_full_image(idx_elegido)
        in_2d, gt_3d = in_2d.to(device), gt_3d.to(device)
        
        pred_3d = modelo(in_2d)
        
        # Calcular métricas del cubo completo
        psnr_val = calcular_psnr(pred_3d, gt_3d)
        ssim_val = calcular_ssim(pred_3d[0], gt_3d[0])
        sam_val = calcular_sam(pred_3d, gt_3d)

    # 4. Preparar tensores para Matplotlib
    img_pan = in_2d[0, 0].cpu().numpy()
    cubo_gt = gt_3d[0].cpu().numpy()     
    cubo_pred = pred_3d[0].cpu().numpy() 
    
    # 5. Configurar el Dashboard Interactivo
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.canvas.manager.set_window_title(f'Inferencia Manual - {nombre_imagen_objetivo}')
    plt.subplots_adjust(bottom=0.25, top=0.85)

    fig.suptitle(f"Imagen: {nombre_imagen_objetivo} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | SAM: {sam_val:.2f}°", 
                 fontsize=14, fontweight='bold')

    longitudes_onda = np.arange(400, 710, 10)
    banda_inicial = 18 

    axes[0].set_title("Medición CASSI PAN")
    axes[0].imshow(img_pan, cmap='gray')
    axes[0].axis('off')

    titulo_gt = axes[1].set_title(f"Verdad Original ({longitudes_onda[banda_inicial]}nm)")
    img_gt_plot = axes[1].imshow(cubo_gt[banda_inicial], cmap='gray', vmin=0, vmax=1)
    axes[1].axis('off')

    titulo_pred = axes[2].set_title(f"Predicción DFHM REAL ({longitudes_onda[banda_inicial]}nm)")
    img_pred_plot = axes[2].imshow(cubo_pred[banda_inicial], cmap='gray', vmin=0, vmax=1)
    axes[2].axis('off')

    error_inicial = np.abs(cubo_gt[banda_inicial] - cubo_pred[banda_inicial])
    axes[3].set_title("Mapa de Error Absoluto")
    img_error_plot = axes[3].imshow(error_inicial, cmap='rainbow', vmin=0, vmax=1) 
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
        
        img_gt_plot.set_data(cubo_gt[b])
        img_pred_plot.set_data(cubo_pred[b])
        img_error_plot.set_data(np.abs(cubo_gt[b] - cubo_pred[b]))
        
        titulo_gt.set_text(f"Verdad Original ({wl}nm)")
        titulo_pred.set_text(f"Predicción DFHM REAL ({wl}nm)")
        
        fig.canvas.draw_idle()

    slider_banda.on_changed(update)
    plt.show()

if __name__ == "__main__":
    main()