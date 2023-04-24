from .clip_transform import (
    clip_transform
)

_transforms = {
    "clip" : clip_transform,
}


def keys_to_transforms(keys: list, size=224):
    return [_transforms[key](size=size) for key in keys]
