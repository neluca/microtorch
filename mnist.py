import numpy as np
import random
import microtorch as mt

from tools import load_mnist, download_mnist
from tqdm import tqdm
import time

EPOCHS = 1
BATCH_SIZE = 32
LR = 4e-3


def one_hot(labels: np.array) -> np.array:
    return np.eye(10)[labels]


def get_batch(images: mt.Tensor, labels: mt.Tensor):
    indices = list(range(0, len(images), BATCH_SIZE))
    random.shuffle(indices)
    for i in indices:
        yield images[i: i + BATCH_SIZE], labels[i: i + BATCH_SIZE]


class Network(mt.Module):
    def __init__(self) -> None:
        self.fc = mt.Linear(28 * 28, 128)
        self.l2 = mt.Linear(128, 10)
        self.ac = mt.Tanh()

    def forward(self, x: mt.Tensor) -> mt.Tensor:
        x = self.ac(self.fc(x))
        return self.l2(x)


@mt.no_grad()
def model_test(model: Network, test_images: mt.Tensor, test_labels: mt.Tensor):
    preds = model.forward(test_images)
    pred_indices = mt.argmax(preds, axis=-1).detach()
    test_labels = test_labels.detach()

    correct = 0
    for p, t in zip(pred_indices.reshape(-1), test_labels.reshape(-1)):
        if p == t:
            correct += 1
    accuracy = correct / len(test_labels)
    print(f"Test accuracy: {accuracy:.2%}")


def train(
        model: Network, optimizer: mt.Adam, train_images: mt.Tensor, train_labels: mt.Tensor
):
    model.train()
    for epoch in range(EPOCHS):
        # Create a tqdm object for the progress bar
        batch_generator = get_batch(train_images, train_labels)
        num_batches = len(train_images) // BATCH_SIZE
        with tqdm(total=num_batches) as pbar:
            for batch_images, batch_labels in batch_generator:
                optimizer.zero_grad()
                pred = model.forward(batch_images)
                loss = mt.cross_entropy(pred, batch_labels)
                loss.backward()
                optimizer.step()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({"loss": float(loss.item())})

        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
        model_test(model, test_images, test_labels)


if __name__ == "__main__":
    download_mnist("mnist")
    (train_images, train_labels), (test_images, test_labels) = load_mnist("mnist")

    train_labels, test_labels = map(mt.tensor, [train_labels, test_labels])

    train_images = mt.tensor(train_images.reshape(-1, 28 * 28) / 255).float()
    test_images = mt.tensor(test_images.reshape(-1, 28 * 28) / 255).float()

    model = Network()
    print("number of parameters: ", model.num_parameters())
    optimizer = mt.Adam(model.parameters(), lr=LR)

    start_time = time.perf_counter()
    train(model, optimizer, train_images, train_labels)
    state_dict = model.state_dict()
    mt.save(model, "mnist_model.mth")

    model2 = Network()
    print(model2)
    mt.load(model2, "mnist_model.mth")

    model_test(model2, test_images, test_labels)

    print(f"Time to train: {time.perf_counter() - start_time} seconds")
