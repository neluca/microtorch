import pickle
from ._modules import Module


def save(model: Module, name: str) -> None:
    with open(name, "wb") as f:
        pickle.dump(model.state_dict(), f)


def load(model: Module, name: str):
    with open(name, "rb") as f:
        state_dict = pickle.load(f)
        model.load_state_dict(state_dict)
