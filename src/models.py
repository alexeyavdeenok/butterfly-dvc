
from tensorflow import keras
from tensorflow.keras import layers


IMAGE_SIZE = (224, 224)
NUM_CLASSES = 100


def build_cnn(
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.3,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> keras.Model:
    """Build a compact CNN trained from scratch.

    Expected input: RGB float32 images with pixel values in [0, 255].
    """
    inputs = keras.Input(shape=(*image_size, 3), name="image")

    # CNN trained from scratch uses pixel values scaled to [0, 1].
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    for filters in (32, 64, 128):
        x = layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    return keras.Model(inputs, outputs, name="simple_cnn")


def build_transfer_model(
    architecture: str,
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.3,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> keras.Model:
    """Build an ImageNet-pretrained model with a frozen backbone.

    Expected input: RGB float32 images with pixel values in [0, 255].
    """
    inputs = keras.Input(shape=(*image_size, 3), name="image")

    if architecture == "mobilenetv2":
        preprocess = keras.applications.mobilenet_v2.preprocess_input

        backbone = keras.applications.MobileNetV2(
            input_shape=(*image_size, 3),
            include_top=False,
            weights="imagenet",
        )

    elif architecture == "efficientnetb0":
        preprocess = keras.applications.efficientnet.preprocess_input

        backbone = keras.applications.EfficientNetB0(
            input_shape=(*image_size, 3),
            include_top=False,
            weights="imagenet",
        )

    else:
        raise ValueError(
            "architecture must be 'mobilenetv2' or 'efficientnetb0'"
        )

    # Train only the new classification head.
    backbone.trainable = False

    # MobileNetV2 scales [0, 255] to [-1, 1].
    # Standard Keras EfficientNetB0 contains its own Rescaling layer;
    # its preprocess_input intentionally leaves inputs unchanged.
    x = preprocess(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    return keras.Model(inputs, outputs, name=architecture)


def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Compile a model for multiclass classification."""
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    for name, builder in (
        ("cnn", lambda: build_cnn()),
        (
            "mobilenetv2",
            lambda: build_transfer_model("mobilenetv2"),
        ),
        (
            "efficientnetb0",
            lambda: build_transfer_model("efficientnetb0"),
        ),
    ):
        model = builder()
        print(f"\n{name}: {model.count_params():,} parameters")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
