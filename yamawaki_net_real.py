import torch
import torch.nn as nn

# 1. El bloque (SHB) fiel a las ecuaciones del paper
class SpectralHallucinationBlock(nn.Module):
    def __init__(self, in_channels, S=2):
        super(SpectralHallucinationBlock, self).__init__()
        self.compressed_channels = in_channels // S
        self.hallucinated_channels = in_channels - self.compressed_channels
        
        # Compresión: El paper especifica kernel 3x3 vanilla (denso)
        self.compress_conv = nn.Conv2d(in_channels, self.compressed_channels, kernel_size=3, padding=1)
        
        # Alucinación: El paper especifica depth-wise (groups = compressed_channels)
        self.hallucinate_conv = nn.Conv2d(self.compressed_channels, self.hallucinated_channels, 
                                          kernel_size=3, padding=1, 
                                          groups=self.compressed_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        intrinsic_features = self.relu(self.compress_conv(x))
        ghost_features = self.relu(self.hallucinate_conv(intrinsic_features))
        out = torch.cat([intrinsic_features, ghost_features], dim=1)
        return out

# 2. El bloque de Atención Espacial (SCAB) - Corregido a Depth-wise
class SpatialContextAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(SpatialContextAttentionBlock, self).__init__()
        # Devolvemos el groups=channels para que sean súper ligeras (Depth-wise)
        self.conv_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv_5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.conv_7x7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        
        self.attention_conv = nn.Conv2d(channels * 3, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale_1 = self.conv_3x3(x)
        scale_2 = self.conv_5x5(x)
        scale_3 = self.conv_7x7(x)
        
        concat = torch.cat([scale_1, scale_2, scale_3], dim=1)
        attention_map = self.sigmoid(self.attention_conv(concat))
        return x * attention_map

# 3. El Módulo Completo (DFHM)
class DFHM(nn.Module):
    def __init__(self, channels, S=2):
        super(DFHM, self).__init__()
        self.shb = SpectralHallucinationBlock(channels, S)
        self.scab = SpatialContextAttentionBlock(channels)

    def forward(self, x):
        residual = x
        out = self.shb(x)
        out = self.scab(out)
        out = out + residual 
        return out

class YamawakiNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=31, num_features=512, num_blocks=9, S=2):
        super(YamawakiNet, self).__init__()
        
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        blocks = [DFHM(num_features, S) for _ in range(num_blocks)]
        self.body = nn.Sequential(*blocks)
        
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        out = self.tail(x)
        return out

# ==========================================
# ZONA DE PRUEBA Y ANÁLISIS PROFUNDO
# ==========================================
def analizar_modelo(modelo, entrada):
    total_params = 0
    trainable_params = 0
    normal_params = 0
    macs_list = []

    # 1. Analizamos los parámetros de cada capa
    for nombre, capa in modelo.named_modules():
        if isinstance(capa, nn.Conv2d):
            # Parámetros reales actuales
            params_capa = sum(p.numel() for p in capa.parameters())
            total_params += params_capa
            
            if capa.weight.requires_grad:
                trainable_params += params_capa

            # Calculamos cómo sería si fuera una convolución "Normal" (groups=1)
            out_c = capa.out_channels
            in_c = capa.in_channels
            k_h, k_w = capa.kernel_size
            
            bias_params = out_c if capa.bias is not None else 0
            normal_params += (out_c * in_c * k_h * k_w) + bias_params

    # 2. Creamos un "espía" (Hook) para contar las operaciones matemáticas (FLOPs)
    def hook_conv(module, input, output):
        out_c = module.out_channels
        in_c = module.in_channels
        k_h, k_w = module.kernel_size
        groups = module.groups
        out_h, out_w = output.shape[2], output.shape[3]
        
        # Fórmula de MACs (Multiply-Accumulate Operations) para Conv2d
        macs = (in_c // groups) * k_h * k_w * out_c * out_h * out_w
        macs_list.append(macs)

    hooks = []
    for capa in modelo.modules():
        if isinstance(capa, nn.Conv2d):
            hooks.append(capa.register_forward_hook(hook_conv))

    # 3. Hacemos una pasada de prueba para activar los espías
    modelo.eval()
    with torch.no_grad():
        modelo(entrada)
        
    # Retiramos los espías para no ensuciar la red
    for h in hooks:
        h.remove()

    total_macs = sum(macs_list)
    # 1 MAC equivale a 2 FLOPs (1 multiplicación + 1 suma)
    total_flops = total_macs * 2 
    
    return total_params, trainable_params, normal_params, total_flops

if __name__ == "__main__":
    # Simulamos el parche comprimido de 48x48
    cassi_snapshot = torch.randn(1, 1, 48, 48) 
    
    # Instanciamos la red con los 512 canales reales de Yamawaki
    modelo_completo = YamawakiNet(in_channels=1, out_channels=31, num_features=512, num_blocks=9, S=2)
    
    # Ejecutamos nuestro escáner
    total, entrenables, normales, flops = analizar_modelo(modelo_completo, cassi_snapshot)
    
    peso_mb = (total * 4) / (1024 ** 2)
    peso_normal_mb = (normales * 4) / (1024 ** 2)
    giga_flops = flops / (10**9)
    
    print(f"--- ANÁLISIS PROFUNDO DE LA RED (YAMAWAKI REAL) ---")
    print(f"Parámetros Totales reales    : {total:,}")
    print(f"Parámetros Entrenables       : {entrenables:,} ({(entrenables/total)*100:.0f}%)")
    print(f"Peso actual del modelo       : {peso_mb:.2f} MB")
    print(f"-"*50)
    print(f"Si usáramos convoluciones NORMALES (sin depth-wise):")
    print(f"Parámetros Normales          : {normales:,}")
    print(f"Peso que tendría el modelo   : {peso_normal_mb:.2f} MB")
    print(f"-"*50)
    print(f"Rendimiento Computacional (Para entrada 48x48):")
    print(f"FLOPs Totales                : {flops:,} operaciones")
    print(f"GigaFLOPs (GFLOPs)           : {giga_flops:.4f} GFLOPs")