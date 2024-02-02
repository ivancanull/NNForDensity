import os
import torch
import numpy as np

from core import *
from utils import *
from torch.utils.tensorboard import SummaryWriter

def train(
    configs: Config,
    writer: SummaryWriter,
    device: torch.device
):
    ckpt_path = os.path.join(configs.model_dir, f'{configs.trial_name}_ckpt.pt')
    best_path = os.path.join(configs.model_dir, f'{configs.trial_name}_best.pt')

    # split dataset
    spliter = Spliter(f"../data/{configs.dataset.name}",
                    configs.dataset.dataset_size,
                    configs.dataset.benchmark)

    spliter.write_file_path_txt()

    # build transforms
    transforms = make_transforms(
        configs=configs
    )

    # build train dataset
    train_dataset = make_datasets(
        configs=configs,
        mode="train",
        train_path=os.path.join(configs.datasets.dir, "train_indices.txt"),
        transforms=transforms,
        device=device
    )
    val_dataset = make_datasets(
        configs=configs,
        mode="val",
        val_path=os.path.join(configs.datasets.dir, "val_indices.txt"),
        transforms=transforms,
        device=device
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs.batch_size
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=configs.batch_size
    )

    # build model
    net = make_model(
        configs=configs
    )
    net.initialize_weights()

    # build optimizer
    optimizer = make_optimizer(
        params=net.parameters(),
        configs=configs,
    )

    # build lr_scheduler
    lr_scheduler = make_lr_scheduler(
        optimizer=optimizer,
        configs=configs,
    )

    # build loss function
    criterion = make_criterion(
        configs=configs,
    )

    # continue training
    if configs.ckpt.read:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            epoch_continue = ckpt["epoch"]
            net.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            transforms = ckpt["transforms"]

    for epoch in range(configs.num_epochs):
        # Define real epoch 

        if epoch < epoch_continue:
            continue
        
        # training
        net.train()
        for i, (X_train, y_train) in enumerate(train_dataloader):

            output = net(X_train)
            train_loss = criterion(output, y_train)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            running_train_loss = train_loss.item() * X_train.size(0)
        
        # validation
        net.eval()
        for i, (X_val, y_val) in enumerate(val_dataloader):
            output = net(X_val)
            val_loss = criterion(output, y_val)
            running_val_loss = val_loss.item() * X_val.size(0)

        train_loss = running_train_loss / len(train_dataloader.dataset)
        val_loss = running_val_loss / len(val_dataloader.dataset)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)


def main():
    # parse arguments
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    configs.trial_name = get_filename(args.setting)[0]

    configs = create_folders(os.getcwd(), configs)

    writer = SummaryWriter(log_dir=f"../runs/{configs.trial_name}")
    
    train(
        configs=configs,
        writer=writer,
        device="cpu"
    )
    writer.flush()
    writer.close()
   

if __name__ == "__main__":
    main()