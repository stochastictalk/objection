from typing import Dict, Optional, Tuple, Union

from torch import nn, Tensor
from torchvision.transforms import functional as F, transforms as T

class PILToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target