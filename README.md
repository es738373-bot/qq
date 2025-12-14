Реализация простой архитектуры U-Net для сегментации изображений
Архитектура U-Net состоит из двух основных частей:
Encoder (сжатие)
Выполняет последовательные свертки и пулы для уменьшения пространственного разрешения и увеличения глубины признаков.
Decoder (расширение)
Восстанавливает разрешение изображения с помощью транспонированных сверток и добавляет skip-connections (пропускающие соединения) для передачи деталей низкого уровня из Encoder-а обратно в Decoder.
Основные элементы реализации:
1. Слои свертки и пулинга (ConvLayer, PoolLayer)
Эти слои отвечают за извлечение признаков высокого уровня и уменьшение размера изображения соответственно.

python
Копировать
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)

class PoolLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        return self.pool(x)
2. Трансформированные свертки для upsampling (UpConvLayer)
Используются для восстановления исходного разрешения изображения.

python
Копировать
class UpConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x):
        return self.upconv(x)
3. Пропускающие соединения (Skip Connections)
Передают низкоуровневые признаки из Encoder-а обратно в соответствующие уровни Decoder-а.

4. Функция потерь Dice Loss
Эта функция используется для оценки схожести предсказанных масок и реальных целевых масок.

python
Копировать
def dice_loss(pred, target):
    smooth = 1.0
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice_score.mean()
Полная реализация класса UNet:
python
Копировать
class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        # Encoder
        self.encode_conv1 = ConvLayer(input_channels, 64)
        self.encode_pool1 = PoolLayer()
        self.encode_conv2 = ConvLayer(64, 128)
        self.encode_pool2 = PoolLayer()
        
        # Bottleneck
        self.bottleneck = ConvLayer(128, 256)
        
        # Decoder
        self.decode_upconv1 = UpConvLayer(256, 128)
        self.decode_conv1 = ConvLayer(256, 128)
        self.decode_upconv2 = UpConvLayer(128, 64)
        self.decode_conv2 = ConvLayer(128, 64)
        
        # Output layer
        self.output_layer = nn.Conv2d(64, output_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoding path
        enc1 = self.encode_conv1(x)
        pool1 = self.encode_pool1(enc1)
        enc2 = self.encode_conv2(pool1)
        pool2 = self.encode_pool2(enc2)
        
        bottleneck = self.bottleneck(pool2)
        
        # Decoding path with skip connections
        up1 = self.decode_upconv1(bottleneck)
        dec1 = torch.cat([up1, enc2], dim=1)
        dec1 = self.decode_conv1(dec1)
        
        up2 = self.decode_upconv2(dec1)
        dec2 = torch.cat([up2, enc1], dim=1)
        dec2 = self.decode_conv2(dec2)
        
        output = self.output_layer(dec2)
        return output
Преобразование любого NFA в эквивалентный DFA
Любой неопредёленный конечный автомат (Nondeterministic Finite Automaton, NFA) может быть преобразован в эквивалентный определённый конечный автомат (Deterministic Finite Automaton, DFA) посредством процедуры, известной как алгоритм построения множеств состояний (или power set construction).

Вот шаги процесса:

Начальное состояние:Стартовым состоянием DFA станет совокупность всех состояний NFA, достижимых по пустым переходам (
ε
ε-переходам) из первоначального состояния NFA.
Переходы:Для каждого текущего состояния DFA (которое представляет собой множество состояний NFA) и для каждого символа входного алфавита создаётся новое состояние путём объединения всех состояний NFA, достижимых из текущих состояний по этому символу, включая 
ε
ε-переходы.
Терминальное состояние:Любое состояние DFA считается терминальным, если оно включает хотя бы одно терминальное состояние исходного NFA.
Продолжаем процесс, создавая новые состояния, пока не будут исчерпаны все возможные комбинации состояний NFA.
Таким образом, создаваемый DFA обеспечивает полную детерминированность, сохраняя способность распознавать тот же самый язык, что и исходный NFA.

Этот метод гарантирует существование эквивалентного DFA для любого заданного NFA.
