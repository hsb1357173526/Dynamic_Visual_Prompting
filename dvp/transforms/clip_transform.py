from torchvision import transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def clip_transform(size=224):
    trs = transforms.Compose([
        transforms.Resize(size, interpolation=BICUBIC),
        transforms.CenterCrop(size),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    return trs
