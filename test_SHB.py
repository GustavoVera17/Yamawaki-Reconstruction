import torch
import torch.nn as nn

class SpectralHallucinationBlock(nn.Module):
    def __init__(self, in_channels, S=2):
        super(SpectralHallucinationBlock, self).__init__()
        self.S = S
        # Calculamos cuántos canales comprimimos y cuántos vamos a "alucinar"
        self.compressed_channels = in_channels // S
        self.hallucinated_channels = in_channels - self.compressed_channels
        
        # 1. Fase de Compresión: Usamos una convolución 1x1 normal (para extraer la esencia)
        self.compress_conv = nn.Conv2d(in_channels, self.compressed_channels, kernel_size=1)
        
        # 2. Fase de Alucinación: Operación "barata" (Depth-wise convolution)
        # El secreto aquí es "groups=self.compressed_channels". Esto obliga a PyTorch a 
        # no mezclar todos los canales, ahorrando millones de operaciones (FLOPs).
        self.hallucinate_conv = nn.Conv2d(self.compressed_channels, self.hallucinated_channels, 
                                          kernel_size=3, padding=1, 
                                          groups=self.compressed_channels)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Extraemos las características intrínsecas (pesadas pero reducidas en número)
        intrinsic_features = self.relu(self.compress_conv(x))
        
        # Generamos las características fantasma a partir de las intrínsecas (muy barato)
        ghost_features = self.relu(self.hallucinate_conv(intrinsic_features))
        
        # Juntamos ambas mitades para restaurar el número original de canales
        out = torch.cat([intrinsic_features, ghost_features], dim=1)
        return out

# ==========================================
# ZONA DE PRUEBA (Sin usar ningún Dataset)
# ==========================================
if __name__ == "__main__":
    # 1. Definimos las dimensiones (Ejemplo: 1 imagen, 64 canales, parches de 48x48)
    # En PyTorch el formato es (Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 64, 48, 48) 
    print(f"Dimensión de entrada: {dummy_input.shape}")

    # 2. Instanciamos el modelo con el factor de compresión S=2 del paper
    modelo_shb = SpectralHallucinationBlock(in_channels=64, S=2)

    # 3. Pasamos el tensor de ruido por el modelo
    salida = modelo_shb(dummy_input)
    print(f"Dimensión de salida : {salida.shape}")

    # 4. Comprobamos si las dimensiones coinciden (Si es True, ¡la arquitectura base funciona!)
    print(f"¿La entrada y salida tienen el mismo tamaño?: {dummy_input.shape == salida.shape}")

    # 5. Contamos los parámetros para ver por qué es tan "ligero"
    params_totales = sum(p.numel() for p in modelo_shb.parameters() if p.requires_grad)
    print(f"Total de parámetros entrenables en este bloque: {params_totales}")
    
    # Solo para comparar, ¿cuánto pesaría una convolución estándar normal de 64 a 64 canales?
    conv_normal = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    params_normal = sum(p.numel() for p in conv_normal.parameters() if p.requires_grad)
    print(f"Parámetros si usaramos una convolución normal : {params_normal}")