import logging
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, TensorDataset


def getImagesDS(X, n):
    image_list = []
    for i in range(n):
        image_list.append(X[i][0].numpy()[None,])
    return np.concatenate(image_list)


def get_num_classes(dataset):
    if hasattr(dataset, "tensors"):
        labels = dataset.tensors[1]
        unique_labels = torch.unique(labels)
        return len(unique_labels)
    else:
        labels = [dataset[i][1] for i in range(min(1000, len(dataset)))]
        return len(set(labels))


def get_dataset(dataset):
    if dataset == "cifar":
        xpriv, xpub = load_cifar()
        print("load cifar success!")
        logging.info("load cifar success!")
    elif dataset == "emnist":
        xpriv, xpub = load_emnist()
        print("load emnist success!")
        logging.info("load emnist success!")
    elif dataset == "fashion":
        xpriv, xpub = load_fashion()
        print("load fashion success!")
        logging.info("load fashion success!")
    elif dataset == "omnist":
        xpriv, xpub = load_omnist()
        print("load omnist success!")
        logging.info("load omnist success!")
    elif dataset == "mnist":
        xpriv, xpub = load_mnist()
        print("load mnist success!")
        logging.info("load mnist success!")
    elif dataset == "stl10":
        xpriv, xpub = load_stl10()
        print("load stl10 success!")
        logging.info("load stl10 success!")
    elif dataset == "tinyimagenet":
        xpriv, xpub = load_tinyimagenet()
        print("load tinyimagenet success!")
        logging.info("load tinyimagenet success!")
    elif dataset == "celeba":
        xpriv, xpub = load_celeba()
        print("load celeba success!")
        logging.info("load celeba success!")
    elif dataset == "utkface":
        xpriv, xpub = load_utkface()
        print("load utkface success!")
        logging.info("load utkface success!")
    else:
        logging.error("Unknown dataset!")
        exit()

    return xpriv, xpub


def load_fashion():
    xpriv = datasets.FashionMNIST(root="./data", train=True, download=True)

    xpub = datasets.FashionMNIST(root="./data", train=False)

    x_train = np.array(xpriv.data)
    y_train = np.array(xpriv.targets)

    x_test = np.array(xpub.data)
    y_test = np.array(xpub.targets)

    x_train = x_train[:, None, :, :]
    x_test = x_test[:, None, :, :]
    x_train = np.tile(x_train, (1, 3, 1, 1))
    x_test = np.tile(x_test, (1, 3, 1, 1))

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    print("x_train.shape in fashion: ", x_train.shape)

    x_train = F.interpolate(x_train, (32, 32))
    x_test = F.interpolate(x_test, (32, 32))
    x_train = x_train / (255 / 2) - 1
    x_test = x_test / (255 / 2) - 1
    x_train = torch.clip(x_train, -1.0, 1.0)
    x_test = torch.clip(x_test, -1.0, 1.0)
    xpriv = TensorDataset(x_train, y_train)
    xpub = TensorDataset(x_test, y_test)
    return xpriv, xpub


def load_omnist():
    xpriv = datasets.Omniglot(
        root="./data", background=True, transform=transforms.ToTensor(), download=True
    )
    xpub = datasets.Omniglot(
        root="./data", background=False, transform=transforms.ToTensor(), download=True
    )

    x_train = np.array([xpriv.__getitem__(i)[0].numpy() for i in range(len(xpriv))])
    y_train = np.array([xpriv.__getitem__(i)[1] for i in range(len(xpriv))])

    x_test = np.array([xpub.__getitem__(i)[0].numpy() for i in range(len(xpub))])
    y_test = np.array([xpub.__getitem__(i)[1] for i in range(len(xpub))])
    print(x_train.shape)

    x_train = np.tile(x_train, (1, 3, 1, 1))
    x_test = np.tile(x_test, (1, 3, 1, 1))

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    print("x_train.shape in omnist: ", x_train.shape)
    x_train = F.interpolate(x_train, (32, 32))
    x_test = F.interpolate(x_test, (32, 32))
    x_train = x_train * 2 - 1
    x_test = x_test * 2 - 1
    x_train = torch.clip(x_train, -1.0, 1.0)
    x_test = torch.clip(x_test, -1.0, 1.0)
    xpriv = TensorDataset(x_train, y_train)
    xpub = TensorDataset(x_test, y_test)
    return xpriv, xpub


def load_mnist():
    xpriv = datasets.MNIST(root="./data", train=True, download=True)

    xpub = datasets.MNIST(root="./data", train=False)

    x_train = np.array(xpriv.data)
    y_train = np.array(xpriv.targets)
    x_test = np.array(xpub.data)
    y_test = np.array(xpub.targets)

    x_train = x_train[:, None, :, :]
    x_test = x_test[:, None, :, :]
    x_train = np.tile(x_train, (1, 3, 1, 1))
    x_test = np.tile(x_test, (1, 3, 1, 1))

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    print("x_train.shape in mnist: ", x_train.shape)
    x_train = F.interpolate(x_train, (32, 32))
    x_test = F.interpolate(x_test, (32, 32))
    x_train = x_train / (255 / 2) - 1
    x_test = x_test / (255 / 2) - 1
    x_train = torch.clip(x_train, -1.0, 1.0)
    x_test = torch.clip(x_test, -1.0, 1.0)
    xpriv = TensorDataset(x_train, y_train)
    xpub = TensorDataset(x_test, y_test)
    return xpriv, xpub


def load_emnist():
    xpriv = datasets.EMNIST(root="./data", train=True, split="letters", download=True)
    xpub = datasets.EMNIST(root="./data", split="letters", train=False, download=True)

    x_train = np.array(xpriv.data)
    y_train = np.array(xpriv.targets) - 1
    x_test = np.array(xpub.data)
    y_test = np.array(xpub.targets) - 1

    x_train = x_train[:, None, :, :]
    x_test = x_test[:, None, :, :]
    x_train = np.tile(x_train, (1, 3, 1, 1))
    x_test = np.tile(x_test, (1, 3, 1, 1))

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train).long()
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test).long()

    print("x_train.shape in emnist: ", x_train.shape)
    x_train = F.interpolate(x_train, (32, 32))
    x_test = F.interpolate(x_test, (32, 32))
    x_train = x_train / (255 / 2) - 1
    x_test = x_test / (255 / 2) - 1
    x_train = torch.clip(x_train, -1.0, 1.0)
    x_test = torch.clip(x_test, -1.0, 1.0)
    xpriv = TensorDataset(x_train, y_train)
    xpub = TensorDataset(x_test, y_test)
    return xpriv, xpub


def load_cifar():
    xpriv = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    xpub = datasets.CIFAR10(
        root="./data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    print("x_train.shape in cifar: ", xpriv.data.shape)
    print("x_test.shape in cifar: ", xpub.data.shape)
    return xpriv, xpub


def load_tinyimagenet():
    import imageio.v2 as imageio

    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, "./data/tiny-imagenet-200/")

    def get_id_dictionary():
        id_dict = {}
        for i, line in enumerate(open(os.path.join(path, "wnids.txt"), "r")):
            id_dict[line.strip()] = i
        return id_dict

    def get_data(id_dict):
        train_data, test_data = [], []
        train_labels, test_labels = [], []

        for key, value in id_dict.items():
            for i in range(500):
                img_path = os.path.join(path, f"train/{key}/images/{key}_{i}.JPEG")
                try:
                    img = imageio.imread(img_path)
                    if len(img.shape) == 2:
                        img = np.stack([img, img, img], axis=-1)

                    elif img.shape[2] == 4:
                        img = img[:, :, :3]

                    if img.shape[:2] != (64, 64):

                        continue

                    train_data.append(img)
                    train_labels.append(value)
                except Exception as e:
                    continue

        val_annotations_path = os.path.join(path, "val/val_annotations.txt")
        for line in open(val_annotations_path):
            img_name, class_id = line.split("\t")[:2]
            img_path = os.path.join(path, f"val/images/{img_name}")
            try:
                img = imageio.imread(img_path)
                if len(img.shape) == 2:
                    img = np.stack([img, img, img], axis=-1)

                elif img.shape[2] == 4:
                    img = img[:, :, :3]

                if img.shape[:2] != (64, 64):

                    continue

                test_data.append(img)
                test_labels.append(id_dict[class_id])
            except Exception as e:
                continue

        return (
            np.array(train_data),
            np.array(train_labels),
            np.array(test_data),
            np.array(test_labels),
        )

    id_dict = get_id_dictionary()
    x_train, y_train, x_test, y_test = get_data(id_dict)

    x_train = torch.Tensor(x_train).permute(0, 3, 1, 2)
    x_test = torch.Tensor(x_test).permute(0, 3, 1, 2)
    x_train = F.interpolate(x_train, (32, 32))
    x_test = F.interpolate(x_test, (32, 32))
    x_train = x_train / (255 / 2) - 1
    x_test = x_test / (255 / 2) - 1
    x_train = torch.clip(x_train, -1.0, 1.0)
    x_test = torch.clip(x_test, -1.0, 1.0)

    y_train = torch.Tensor(y_train).long()
    y_test = torch.Tensor(y_test).long()

    print("x_train.shape: ", x_train.shape)
    print("x_test.shape: ", x_test.shape)

    xpriv = TensorDataset(x_train, y_train)
    xpub = TensorDataset(x_test, y_test)
    return xpriv, xpub


def load_stl10():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, "./data/stl10_binary/")

    with open(os.path.join(path, "train_X.bin"), "rb") as f:
        train_data = np.fromfile(f, dtype=np.uint8)
        train_data = np.reshape(train_data, (-1, 3, 96, 96))
    with open(os.path.join(path, "test_X.bin"), "rb") as f:
        test_data = np.fromfile(f, dtype=np.uint8)
        test_data = np.reshape(test_data, (-1, 3, 96, 96))
    with open(os.path.join(path, "train_y.bin"), "rb") as f:
        train_labels = np.fromfile(f, dtype=np.uint8).reshape(-1)
    with open(os.path.join(path, "test_y.bin"), "rb") as f:
        test_labels = np.fromfile(f, dtype=np.uint8).reshape(-1)

    import torch
    import torch.nn.functional as F

    x_train = torch.Tensor(train_data)
    x_test = torch.Tensor(test_data)
    print("x_train.shape in stl10: ", x_train.shape)
    x_train = F.interpolate(x_train, (32, 32))
    x_test = F.interpolate(x_test, (32, 32))
    x_train = x_train / (255 / 2) - 1
    x_test = x_test / (255 / 2) - 1
    x_train = torch.clip(x_train, -1.0, 1.0)
    x_test = torch.clip(x_test, -1.0, 1.0)

    y_train = torch.Tensor(train_labels).long() - 1
    y_test = torch.Tensor(test_labels).long() - 1

    xpriv = TensorDataset(x_train, y_train)
    xpub = TensorDataset(x_test, y_test)
    return xpriv, xpub


def load_celeba():

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    try:
        celeba_train = datasets.CelebA(
            root="./data",
            split="train",
            target_type="attr",
            transform=transform,
            download=True,
        )

        celeba_test = datasets.CelebA(
            root="./data",
            split="test",
            target_type="attr",
            transform=transform,
            download=False,
        )

    except Exception as e:
        print(
            "Please manually download the CelebA dataset to the ./data/celeba directory"
        )
        raise e

    x_train_list = []
    y_train_list = []

    train_samples = len(celeba_train)
    for i in range(train_samples):
        if i % 1000 == 0:
            print(f"Processing train samples: {i}/{train_samples}")
        img, attr = celeba_train[i]
        x_train_list.append(img.numpy())
        label = attr[20].item()  # male attribute
        y_train_list.append(label)

    x_test_list = []
    y_test_list = []

    test_samples = len(celeba_test)
    for i in range(test_samples):
        if i % 500 == 0:
            print(f"Processing test samples: {i}/{test_samples}")
        img, attr = celeba_test[i]
        x_test_list.append(img.numpy())
        label = attr[20].item()
        y_test_list.append(label)

    x_train = np.array(x_train_list)
    y_train = np.array(y_train_list)
    x_test = np.array(x_test_list)
    y_test = np.array(y_test_list)

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    x_train = F.interpolate(x_train, (32, 32))
    x_test = F.interpolate(x_test, (32, 32))

    x_train = x_train * 2 - 1
    x_test = x_test * 2 - 1
    x_train = torch.clip(x_train, -1.0, 1.0)
    x_test = torch.clip(x_test, -1.0, 1.0)

    xpriv = TensorDataset(x_train, y_train)
    xpub = TensorDataset(x_test, y_test)

    return xpriv, xpub


class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, task="gender"):
        """
        Args:
            root_dir: UTKFace dir
            task: 'gender', 'age', 'race'
        """
        self.root_dir = root_dir
        self.task = task
        self.image_files = []
        self.labels = []

        for filename in os.listdir(root_dir):
            if filename.endswith(".jpg"):
                try:
                    parts = filename.split("_")
                    age = int(parts[0])
                    gender = int(parts[1])
                    race = int(parts[2])

                    self.image_files.append(os.path.join(root_dir, filename))

                    if task == "gender":
                        self.labels.append(gender)
                    elif task == "age":
                        if age <= 20:
                            age_group = 0
                        elif age <= 40:
                            age_group = 1
                        elif age <= 60:
                            age_group = 2
                        else:
                            age_group = 3
                        self.labels.append(age_group)
                    elif task == "race":
                        self.labels.append(race)

                except (ValueError, IndexError):
                    continue

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        return image, label


def load_utkface():
    utkface_path = "./data/UTKFace/all"

    if not os.path.exists(utkface_path):
        print("UTKFace dataset not found!")
        print("Please download the UTKFace dataset from the following link:")
        print("https://susanqq.github.io/UTKFace/")
        print("And extract it to the ./data/UTKFace/ directory")
        raise FileNotFoundError("UTKFace dataset not found")

    full_dataset = UTKFaceDataset(utkface_path, task="gender")

    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)

    images_list = []
    labels_list = []

    from PIL import Image

    target_size = (200, 200)

    for i in range(total_len):
        if i % 1000 == 0:
            print(f"Processing samples: {i}/{total_len}")

        try:
            image, label = full_dataset[i]

            image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
            image_np = np.array(image_resized)

            if len(image_np.shape) == 2:
                image_np = np.stack([image_np, image_np, image_np], axis=-1)
            elif image_np.shape[2] == 4:
                image_np = image_np[:, :, :3]

            if image_np.shape[:2] == target_size:
                images_list.append(image_np)
                labels_list.append(label)
            else:
                print(f"Skipping image with abnormal size: {image_np.shape}")

        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue

    print(f"Successfully processed {len(images_list)} images")

    # Now all images have the same size, we can safely create numpy arrays
    images = np.array(images_list)
    labels = np.array(labels_list)

    total_processed = len(images)
    train_len = int(0.8 * total_processed)

    train_images = images[:train_len]
    train_labels = labels[:train_len]
    test_images = images[train_len:]
    test_labels = labels[train_len:]

    x_train = torch.Tensor(train_images).permute(0, 3, 1, 2)
    x_test = torch.Tensor(test_images).permute(0, 3, 1, 2)

    x_train = F.interpolate(x_train, (32, 32))
    x_test = F.interpolate(x_test, (32, 32))

    x_train = x_train / (255 / 2) - 1
    x_test = x_test / (255 / 2) - 1
    x_train = torch.clip(x_train, -1.0, 1.0)
    x_test = torch.clip(x_test, -1.0, 1.0)

    y_train = torch.Tensor(train_labels).long()
    y_test = torch.Tensor(test_labels).long()

    xpriv = TensorDataset(x_train, y_train)
    xpub = TensorDataset(x_test, y_test)

    return xpriv, xpub
