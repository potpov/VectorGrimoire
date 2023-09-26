from matplotlib import pyplot as plt
import torch
from torch import nn
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
from torchvision import transforms, datasets
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import glob
import csv
import re

from thesis.models.autoencoder import AutoEncoder
from thesis.dataset import EmojiDataset, CausalSVGDataModule

OUTPUT_DIR = "/scratch2/moritz_logs/AE_fc_res18"
CONTINUE_TRAINING = False
ENCODER_STRING = "resnet18"
USE_FULLY_CONNECTED_DECODER = True

continue_epoch = None

if not os.path.exists(OUTPUT_DIR):
    if CONTINUE_TRAINING:
        print(f"[WARNING] Cannot resume training, >> {OUTPUT_DIR} << is not a dir.")
    print("Creating new output directory... ")
    os.mkdir(OUTPUT_DIR)
    continue_epoch = 0
else:
    checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "*.pth"))
    if CONTINUE_TRAINING and len(checkpoints) > 0:
        checkpoints.sort(key=os.path.getmtime)
        latest_checkpoint = checkpoints[-1]
        print(f"Resuming training from checkpoint {latest_checkpoint}")
        model = AutoEncoder(ENCODER_STRING, use_fc = False).cuda()
        model.load_state_dict(torch.load(latest_checkpoint))
        pattern = r".*(\d+).*\.pth"
        match = re.search(pattern, latest_checkpoint)
        continue_epoch = match.group(1)
    else:
        input(f"Confirm to delete all old files and checkpoints in {OUTPUT_DIR}...")
        continue_epoch = 0
        print("Removing old files...")
        [os.remove(f) for f in glob.glob(os.path.join(OUTPUT_DIR, "*"))]

# Create a CSV file for logging
csv_file = open(os.path.join(OUTPUT_DIR, 'loss_log.csv'), 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Epoch', 'Batch', 'Loss'])  # Write header row

num_epochs = 350
batch_size = 1  # batch size of 1 is fine as we also have the time dimension
learning_rate = 3e-4

# dataset = EmojiDataset("/home/mfeuerpfeil/master/thesis/datasets/openmoji_faces",
#                           train_batch_size=batch_size,
#                           val_batch_size=batch_size,
#                           patch_size=128)
# dataset.setup()
# dataloader = dataset.train_dataloader()

dataset = CausalSVGDataModule("/scratch2/moritz_data/causal_openmoji_black_fixed/length",
                              train_batch_size=batch_size,
                              val_batch_size=batch_size,
                              width=128,
                              context_length=27,
                              channels=3)
dataset.setup()
dataloader = dataset.train_dataloader()

model = AutoEncoder(ENCODER_STRING, use_fc = USE_FULLY_CONNECTED_DECODER).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()

for epoch in range(num_epochs):
    if continue_epoch > 0 and epoch <= int(continue_epoch):
        continue
    total_loss = 0
    for i, data in enumerate(dataloader):
        images, shape_layers, stop_signals, caption, merged_layers, merged_images = data
        bs, t, c, w, h = shape_layers.shape
        mask = stop_signals == 0.
        mask = mask.view(*mask.shape, 1, 1, 1)  # ensure broadcasting
        selected_gt_shape_layers = torch.masked_select(shape_layers, mask).view(-1, c, w, h)
        selected_gt_merged_layers = torch.masked_select(merged_layers, mask).view(-1, c, w, h)

        # shape_layers = shape_layers.view(-1, 3, 128, 128)
        # merged_layers = merged_layers.view(-1, 3, 128, 128)
        img = torch.cat([selected_gt_shape_layers, selected_gt_merged_layers], dim=0).cuda()
        
        # print(img.shape)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        print('epoch [{}/{}] step [{}/{}] avg loss:{:.4f}'.format(epoch+1, num_epochs, i+1, len(dataloader), total_loss / i))

        if i % 20 == 0:
            batch_loss = loss.item()
            csv_writer.writerow([epoch, i, batch_loss])

        if i % 100 == 0:
            csv_file.flush()
            rand_idx = torch.randint(1, img.shape[0], (2,))
            output = output.detach().cpu()
            img = img.cpu()
            grid = make_grid([img[0], output[0], img[rand_idx[0]], output[rand_idx[0]],  img[rand_idx[1]], output[rand_idx[1]]], nrow=2, pad_value = 0.1)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(grid.permute(1, 2, 0))
            ax.set_title(f"Epoch: {epoch}, Iteration: {i}/{len(dataloader)}")
            fig.savefig(os.path.join(OUTPUT_DIR, f'plot_{epoch}_{i}.png'))
            # save_image(output[0].detach(), os.path.join(OUTPUT_DIR, f'image_{epoch}_{i}.png'))
            # save_image(img[0].detach(), os.path.join(OUTPUT_DIR, f'image_{epoch}_{i}_input.png'))

    # save the model and delete any model older than 5 epochs
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'./openmoji_autoencoder_epoch_{epoch}.pth'))
    if os.path.exists(os.path.join(OUTPUT_DIR, f'./openmoji_autoencoder_epoch_{epoch-5}.pth')):
        os.remove(os.path.join(OUTPUT_DIR, f'./openmoji_autoencoder_epoch_{epoch-5}.pth'))
    
    # ===================log========================
    print('epoch [{}/{}], total loss:{:.4f}'
          .format(epoch+1, num_epochs, total_loss))
    # if epoch % 5 == 0:
    # save_image(output[0], os.path.join(OUTPUT_DIR, 'image_{}.png'.format(epoch)))
csv_file.close()
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, './openmoji_autoencoder.pth'))