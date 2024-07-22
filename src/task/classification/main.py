import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, tqdm, random


# Define the Transformer model
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size, nhead=1, num_encoder_layers=1, num_decoder_layers=1
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)  # Shape: (seq_len, batch_size, hidden_dim)
        output = self.transformer(embedded, embedded)
        output = output.permute(1, 0, 2)  # Shape: (batch_size, seq_len, hidden_dim)
        output = output.mean(dim=1)  # Shape: (batch_size, hidden_dim)
        logits = self.fc(output)
        return logits


all_data = []
validation_data = []
class_name = {}
class_name_str = []

minimum_length = 10000


def process_files_in_folder(folder_path):
    global all_data
    global minimum_length
    file_name_list = os.listdir(folder_path)
    file_name_list.sort()

    for file_name in file_name_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".npy"):
            data = np.load(file_path)
            if len(data) < minimum_length:
                minimum_length = len(data)

            all_data.append([data, folder_path])

    class_name[folder_path] = len(class_name)
    class_name_str.append(folder_path)


def process_subfolders(parent_folder):
    subfolder_list = os.listdir(parent_folder)
    subfolder_list.sort()
    for subfolder in subfolder_list:
        subfolder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(subfolder_path):
            process_files_in_folder(subfolder_path)


class_idx = 0


def valid_process_files_in_folder(folder_path):
    global validation_data
    global minimum_length
    global class_idx
    file_name_list = os.listdir(folder_path)
    file_name_list.sort()
    for file_name in file_name_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".npy"):
            data = np.load(file_path)
            validation_data.append([data, class_name_str[class_idx]])
            print(file_path, class_name_str[class_idx])


def valid_process_subfolders(parent_folder):
    global class_idx
    subfolder_list = os.listdir(parent_folder)
    subfolder_list.sort()

    for subfolder in subfolder_list:
        subfolder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(subfolder_path):
            valid_process_files_in_folder(subfolder_path)
        class_idx += 1


from same.test import save_bvh_z

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_epoch", default="ckpt0", type=str)
    parser.add_argument("--data", default="classification", type=str)
    parser.add_argument("--out", default="nn", type=str)
    parser.add_argument("--reprocess", action="store_true")
    args = parser.parse_args()

    import os
    from mypath import DATA_DIR

    ## first convert bvh to z(SAME space latent vector)
    # save at [data]/train(or validate)/[model]/[category]/*.npy,
    # so we can process once, and load it later
    train_bvh_dir = os.path.join(DATA_DIR, args.data, "train", "bvh")
    train_npy_dir = os.path.join(DATA_DIR, args.data, "train", args.model_epoch)
    val_bvh_dir = os.path.join(DATA_DIR, args.data, "validate", "bvh")
    val_npy_dir = os.path.join(DATA_DIR, args.data, "validate", args.model_epoch)
    if args.reprocess or (not os.path.exists(train_npy_dir)):
        save_bvh_z(args.model_epoch, train_bvh_dir, train_npy_dir)
    if args.reprocess or (not os.path.exists(val_npy_dir)):
        save_bvh_z(args.model_epoch, val_bvh_dir, val_npy_dir)

    save_dir = os.path.join(DATA_DIR, args.data, "nn", args.model_epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load data
    process_subfolders(train_npy_dir)
    valid_process_subfolders(val_npy_dir)

    random.shuffle(all_data)
    input_size = len(all_data[0][0][0])  # Input size of each time step
    hidden_size = 64  # Number of hidden units in the Transformer
    num_classes = len(class_name)  # Number of output classes
    learning_rate = 0.0005
    num_epochs = 200  # 000
    model = TransformerClassifier(input_size, hidden_size, num_classes).cuda()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(num_epochs):

        random.shuffle(all_data)

        train_data = []
        labels = []
        idx = 0
        for d in all_data:
            train_data.append(np.array([d[0]], dtype=np.float32))
            label = np.zeros(len(class_name))
            label[int(class_name[d[1]])] = 1.0
            labels.append(label)
            idx += 1

        labels = torch.tensor(labels, device="cuda", dtype=torch.float32)

        loss = 0
        mini_batch_size = len(all_data) // 4
        outputs = []
        cnt = 0

        counter = {}

        for i in range(len(train_data)):
            train_in = torch.tensor(train_data[i], device="cuda", dtype=torch.float32)
            # embed()
            outputs.append(model(train_in)[0])
            if torch.argmax(outputs[-1]) == torch.argmax(labels[i]):
                cnt += 1
            if i != 0 and i % mini_batch_size == mini_batch_size - 1:
                outputs = torch.stack(outputs)

                loss = 10.0 * criterion(
                    outputs, labels[i - (mini_batch_size - 1) : i + 1]
                )
                loss /= mini_batch_size

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                outputs = []

        validation_cnt = 0
        with torch.no_grad():
            for data in validation_data:
                out = model(torch.tensor([data[0]], device="cuda", dtype=torch.float32))
                if torch.argmax(out) == class_name[data[1]]:
                    validation_cnt += 1
                else:
                    print(data[1], "(", class_name_str[torch.argmax(out)], ")", end=" ")
            print()

        torch.save(model.state_dict(), os.path.join(save_dir, str(epoch)))
        print(
            "Epoch [{}/{}] [{}], Loss: {:.8f}, Accuracy : {} Validation Accuracy {}/{}".format(
                epoch + 1,
                num_epochs,
                i,
                loss.item(),
                cnt,
                validation_cnt,
                len(validation_data),
            )
        )
