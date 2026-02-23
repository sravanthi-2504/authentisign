def __init__(self):
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "model",
        "signature_embedding_model.keras"
    )

    self.model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False,
        custom_objects={"L2Normalization": L2Normalization}
    )