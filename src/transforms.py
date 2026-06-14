import albumentations as A


def get_train_transforms():

    return A.Compose(

        [
            A.Resize(512, 512),

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.Rotate(
                limit=30,
                p=0.5
            ),

            A.RandomBrightnessContrast(
                p=0.3
            ),

            A.GaussianBlur(
                p=0.2
            )
        ],

        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False
        )
    )


def get_val_transforms():

    return A.Compose(

        [
            A.Resize(512, 512)
        ],

        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False
        )
    )