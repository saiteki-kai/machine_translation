from comet import download_model, load_from_checkpoint
from comet.models import CometModel


def load_comet(model_name: str) -> CometModel:
    model = download_model(model_name)

    return load_from_checkpoint(model)


def comet_score(model: CometModel, data: list[dict[str, str]], batch_size: int) -> tuple[list[float], float]:
    model.set_embedding_cache()
    model.eval()

    outputs = model.predict(data, batch_size=batch_size)

    return outputs["scores"], outputs["system_score"]
