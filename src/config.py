config = {

    "experiment_name": "CNNClassifier",

    "model_name": "CNNClassifier_89.73",

    "experiment_dir": "../experiments/",

    "random_seed": 22,

    "training": {
        "batch_size": 128,
        "learning_rate": 0.001,
        "epochs": 50,
        "loss_function": "cross_entropy",

        "optimizer": {
            "type": "adam",
            "weight_decay": 1e-4,
            "momentum": None
        },
        "lr_scheduler": {
            "type": "plateau",
            "factor": 0.5,
            "patience": 2,
            "t_max": None
        }
    },

    "augmentation": {
        "type": {
            "random_horizontal_flip": True,
            "random_rotation": {
                "enabled": True,
                "degrees": 15
            },
            "random_crop": {
                "enabled": True,
                "size": 32,
                "padding": 4
            },
            "color_jitter": {
                "enabled": True,
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.2
            },
            "normalization": {
                "mean": (0.4914, 0.4822, 0.4465),
                "std": (0.2023, 0.1994, 0.2010)
            }
        }
    },

    "data": {
        "directory": "../data",
        "num_workers": 0,
        "validation_split": 0.2
    },
}