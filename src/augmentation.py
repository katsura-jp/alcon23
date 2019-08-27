import albumentations as albu
from albumentations.pytorch import ToTensor


def get_train_augmentation(height=336, width=224, scale_height=384, scale_width=256):
    train_augmentation = albu.Compose([
        # albu.HorizontalFlip(),
        albu.OneOf([
            albu.RandomBrightness(0.1, p=1),
            albu.RandomContrast(0.1, p=1)
        ], p=0.9),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3),
        albu.OneOf([
            albu.IAAAdditiveGaussianNoise(p=1.0),
            albu.GaussNoise(var_limit=(10., 50.), p=1),
        ], p=0.7),
        albu.OneOf([
            albu.Compose([
                albu.Resize(scale_height, scale_width, p=1.0),
                albu.RandomCrop(height=height, width=width, p=1.0),
            ]),
            albu.Resize(height, width, p=1.0),
        ], p=1.0),
        albu.MotionBlur(blur_limit=10, p=0.7),
        albu.Normalize(mean=[0.695, 0.658, 0.592], std=[0.191, 0.185, 0.171], max_pixel_value=255),
        # albu.Cutout(num_holes=1, max_h_size=height//3, max_w_size=width//3, p=0.5),
        ToTensor()
    ])
    return train_augmentation


def get_test_augmentation(height=336, width=224, scale_height=384, scale_width=256):
    test_augmentation = albu.Compose([
        albu.Resize(height, width, p=1.0),
        albu.Normalize(mean=[0.695, 0.658, 0.592], std=[0.191, 0.185, 0.171], max_pixel_value=255),
        ToTensor()
    ])
    return test_augmentation


