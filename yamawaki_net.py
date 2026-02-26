import torch
import torch.nn as nn

# 1. El bloque que ya probaste y funciona (SHB)
class SpectralHallucinationBlock(nn.Module):
    def __init__(self, in_channels, S=2):
        super(SpectralHallucinationBlock, self).__init__()
        self.compressed_channels = in_channels // S
        self.hallucinated_channels = in_channels - self.compressed_channels
        
        # Compresión
        self.compress_conv = nn.Conv2d(in_channels, self.compressed_channels, kernel_size=1)
        # Alucinación (Operación barata)
        self.hallucinate_conv = nn.Conv2d(self.compressed_channels, self.hallucinated_channels, 
                                          kernel_size=3, padding=1, 
                                          groups=self.compressed_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        intrinsic_features = self.relu(self.compress_conv(x))
        ghost_features = self.relu(self.hallucinate_conv(intrinsic_features))
        out = torch.cat([intrinsic_features, ghost_features], dim=1)
        return out

# 2. El nuevo bloque de Atención Espacial (SCAB)
class SpatialContextAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(SpatialContextAttentionBlock, self).__init__()
        # Usamos operaciones muy baratas para extraer contexto espacial a diferentes escalas
        self.conv_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv_5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.conv_7x7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        
        # Reducción para crear el "mapa de atención"
        self.attention_conv = nn.Conv2d(channels * 3, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Extraemos contexto a escala pequeña (3x3) y mediana (5x5)
        scale_1 = self.conv_3x3(x)
        scale_2 = self.conv_5x5(x)
        scale_3 = self.conv_7x7(x)
        
        # Unimos y calculamos qué píxeles son más importantes (Atención)
        concat = torch.cat([scale_1, scale_2, scale_3], dim=1)
        attention_map = self.sigmoid(self.attention_conv(concat))
        
        # Multiplicamos la entrada original por el mapa de atención para resaltar los bordes
        return x * attention_map

# 3. El Módulo Completo (DFHM = SHB + SCAB)
class DFHM(nn.Module):
    def __init__(self, channels, S=2):
        super(DFHM, self).__init__()
        # La arquitectura de Yamawaki usa conexiones residuales
        self.shb = SpectralHallucinationBlock(channels, S)
        self.scab = SpatialContextAttentionBlock(channels)

    def forward(self, x):
        residual = x
        out = self.shb(x)
        out = self.scab(out)
        # Sumamos la entrada original al final (Residual Connection) para estabilidad
        out = out + residual 
        return out

# ==========================================
# ZONA DE PRUEBA DEL DFHM COMPLETO
# ==========================================
if __name__ == "__main__":
    # Simulamos el parche de 48x48
    dummy_input = torch.randn(1, 64, 48, 48) 
    
    # Instanciamos el módulo completo
    modelo_dfhm = DFHM(channels=64, S=2)
    salida = modelo_dfhm(dummy_input)
    
    params_dfhm = sum(p.numel() for p in modelo_dfhm.parameters() if p.requires_grad)
    
    print(f"--- PRUEBA DEL DFHM COMPLETO ---")
    print(f"Dimensión de entrada : {dummy_input.shape}")
    print(f"Dimensión de salida  : {salida.shape}")
    print(f"Total de parámetros del DFHM completo: {params_dfhm}")
class YamawakiNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=31, num_features=64, num_blocks=9, S=2):
        super(YamawakiNet, self).__init__()
        
        # 1. Cabeza: De la imagen comprimida (1 canal) a 64 canales de características
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        # 2. Cuerpo: Apilamos los 9 módulos DFHM usando nn.Sequential
        # Esto crea una "tubería" donde la salida de un bloque es la entrada del siguiente
        blocks = [DFHM(num_features, S) for _ in range(num_blocks)]
        self.body = nn.Sequential(*blocks)
        
        # 3. Cola: De los 64 canales de características al cubo HSI final (31 canales)
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Flujo de datos a través de la red
        x = self.head(x)
        x = self.body(x)
        out = self.tail(x)
        return out

# ==========================================
# ZONA DE PRUEBA DE LA RED COMPLETA
# ==========================================
if __name__ == "__main__":
    # Simulamos el parche comprimido que entregaría la cámara CASSI:
    # (1 imagen, 1 canal de luz comprimida, 48x48 píxeles espaciales)
    cassi_snapshot = torch.randn(1, 1, 48, 48) 
    
    # Instanciamos la red completa
    modelo_completo = YamawakiNet(in_channels=1, out_channels=29, num_features=64, num_blocks=9, S=2)
    
    # Pasamos la imagen comprimida por toda la red
    hsi_reconstruido = modelo_completo(cassi_snapshot)
    
    # Calculamos el tamaño del modelo en Megabytes
    # Cada parámetro en PyTorch (float32) ocupa 4 bytes
    params_totales = sum(p.numel() for p in modelo_completo.parameters() if p.requires_grad)
    peso_mb = (params_totales * 4) / (1024 ** 2)
    
    print(f"--- PRUEBA DE LA RED COMPLETA (YAMAWAKI) ---")
    print(f"Entrada (Snapshot CASSI) : {cassi_snapshot.shape}")
    print(f"Salida (Cubo HSI 31 bandas): {hsi_reconstruido.shape}")
    print(f"Parámetros Totales       : {params_totales:,}")
    print(f"Peso Estimado del Modelo : {peso_mb:.2f} MB")