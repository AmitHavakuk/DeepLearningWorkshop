import math
import os

from MobileFaceNets import MobileFaceNet
from SupConLoss import SupConLoss
import numpy as np
import torch
from torch.nn import Module, Parameter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from pytorch_metric_learning import miners, distances
from pytorch_metric_learning.losses import TripletMarginLoss
from PIL import Image
import cv2
import itertools
from itertools import chain
import time
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer
from sklearn.metrics import roc_curve
import shutil
import random

#IMAGE_SIZE = (160,160)  # Image dimensions as input to facenet-pytorch
# IMAGE_SIZE = (178,218)  # Image dimensions as input to MySimpleResnet

#IMAGE_SIZE = (224, 224)  # Image dimensions as input to ResNetEmbedding
IMAGE_SIZE = (112, 112)  # Image dimensions as input to MobileFaceNets

IMGS_PER_ID_TYPE = 2
IMG_TYPES = ["natural", "sunglasses", "mask", "augmented_natural", "augmented_sunglasses", "augmented_mask"]
# IMG_TYPES = ["natural", "sunglasses", "augmented_natural", "augmented_sunglasses"]
IMGS_PER_ID = IMGS_PER_ID_TYPE * len(IMG_TYPES)  # dont change
NUM_CLASSES = 10177  # this later changes to exact len of train identities


# Define Hyperparameters
DISTANCE_FUNC_NAME = "cosine"  # key in dictionary

TRAIN_SIZE = 20000
VAL_SIZE = 1024
EMBEDDING_DIM = 512  # Dimension of output embedding
BATCH_SIZE = 16  # Batch size for training
BATCH_TRIOS = (BATCH_SIZE * IMGS_PER_ID) * (IMGS_PER_ID - 1) * ((BATCH_SIZE - 1) * IMGS_PER_ID)  # number of trios considered in a batch
# LEARNING_RATE = 1e-4  # Learning rate for the optimizer
LEARNING_RATE = 1e-3
EPOCHS = 40  # Number of training epochs
# MARGIN = 0.2  # Margin for triplet loss
MARGIN = 0.2

dataset_path = r'./datasets/0_identity'
val_dataset_path = r'./datasets/1_identity'
test_dataset_path = r'./datasets/2_identity'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda")

DISTANCE_FUNC_DICT = {"euc": distances.LpDistance(p=2),
                      "cosine": distances.CosineSimilarity()}
# DISTANCE_FUNC_DICT = {"euc": F.pairwise_distance, "cosine": distances.CosineSimilarity()}

DISTANCE_FUNC = DISTANCE_FUNC_DICT[DISTANCE_FUNC_NAME]
# DISTANCE_FUNC = F.pairwise_distance
loss_func = TripletMarginLoss(margin=MARGIN, distance=DISTANCE_FUNC)
# loss_func = SupConLoss(temperature=0.07, contrast_mode='all',
#                  base_temperature=0.07, device='cuda:0').to(device)
miner = miners.TripletMarginMiner(margin=MARGIN, type_of_triplets="all",
                                  distance=DISTANCE_FUNC)

criterion = torch.nn.CrossEntropyLoss()



class ArcFace(Module):
    """Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """
    def __init__(self, feat_dim, num_class, margin_arc=0.35, margin_am=0.0, scale=32):
        super(ArcFace, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin_arc = margin_arc
        self.margin_am = margin_am
        self.scale = scale
        self.cos_margin = math.cos(margin_arc)
        self.sin_margin = math.sin(margin_arc)
        self.min_cos_theta = math.cos(math.pi - margin_arc)

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_margin - sin_theta * self.sin_margin
        # 0 <= theta + m <= pi, ==> -m <= theta <= pi-m
        # because 0<=theta<=pi, so, we just have to keep theta <= pi-m, that is cos_theta >= cos(pi-m)
        cos_theta_m = torch.where(cos_theta > self.min_cos_theta, cos_theta_m, cos_theta-self.margin_am)
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.byte().bool()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        return output


class CosFace(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def forward(self, input, label=None):
        # cosine = self.cosine_sim(input, self.weight).clamp(-1,1)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(
            -1, 1)

        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)

        return output  # , F.normalize(self.weight, p=2, dim=1), (cosine * one_hot)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'

    def cosine_sim(self, x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1, w2).clamp(min=eps)




def compute_eer(fpr,tpr):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    eer = np.around(eer, 4)
    return eer, min_index


def image_crop_to_tensor(cv2_img):
    # Convert OpenCV BGR image to RGB
    cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    tmp_image = Image.fromarray(cv2_img_rgb)

    # Apply transformations
    tmp_image = transform(tmp_image)  # Now in (C, H, W) format

    return tmp_image


def image_path_to_tensor(image_path):
    # Open image and convert to RGB
    tmp_image = Image.open(image_path).convert('RGB')

    # Apply transformations
    tmp_image = transform(tmp_image)  # Now in (C, H, W) format

    return tmp_image

# below function is deprecated
# def image_path_to_tensor(image_path):
#     tmp_image = Image.open(image_path).convert('RGB')
#     #tmp_image.resize(IMAGE_SIZE)  # DELETE THIS?
#     # Convert to NumPy array
#     tmp_image = np.array(tmp_image)
#
#     # Convert to PyTorch tensor
#     tmp_image = torch.tensor(tmp_image, dtype=torch.float32)
#
#     # Rearrange dimensions from HWC to NCHW
#     tmp_image = tmp_image.permute(2, 0, 1)
#
#     return tmp_image


# Define the transformations for data augmentation
transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # Standard normalization for FaceNet and MobileFaceNet
])

#
# transform = transforms.Compose([
#     transforms.Resize((178,218)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     # Standard normalization for ResNet
# ])


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, size=1000, test=False):
        self.size = size
        self.test = test
        self.img_dir = root_dir
        self.transform = transform
        # Store only paths, not the images themselves
        self.identity_paths_labels = []
        sorted_identities = sorted(os.listdir(root_dir),
                                   key=lambda x: int(os.path.splitext(x)[0]),
                                   reverse=self.test)

        for idx, identity in enumerate(sorted_identities):
            if idx >= self.size:
                break
            #int_id = int(identity)-1
            int_id = idx  # so that labels are consecutive
            if int(identity) % 100 == 0:
                print("Loading identity: ", identity)
            identity_path = os.path.join(root_dir, identity)
            self.identity_paths_labels.append((identity_path, int_id))
        self.size = len(self.identity_paths_labels)

    def __len__(self):
        return len(self.identity_paths_labels)

    def __getitem__(self, idx):
        pictures = []
        identity_path, identity_label = self.identity_paths_labels[idx]
        for folder_type in os.listdir(identity_path):
            if folder_type not in IMG_TYPES:
                continue
            folder_path = os.path.join(identity_path, folder_type)
            with os.scandir(folder_path) as entries:
                random_images_paths = []
                files_paths = [entry.path for entry in entries]
                if len(files_paths) == 0:
                    continue
                elif len(files_paths) < IMGS_PER_ID_TYPE:
                    random_images_paths = files_paths
                else:
                    # print("A", files_paths)
                    # Choose 2 unique random files from the list
                    random_images_paths = random.sample(files_paths,
                                                        IMGS_PER_ID_TYPE)
                    # print("B", random_images_paths)
                for img_path in random_images_paths:
                    # print("C", img_path)
                    tensor_image = image_path_to_tensor(img_path)
                    # if self.transform:
                    #     tensor_image = self.transform(tensor_image)
                    pictures.append(tensor_image)

        if len(pictures) != IMGS_PER_ID:
            #print("BAD len:", len(pictures), "identity: ", identity_label)
            for i in range(IMGS_PER_ID - len(
                    pictures)):  # Cheap fix to make total IMG_PER_ID pictures
                pictures.append(random.choice(pictures))

        pictures = torch.stack(pictures)
        return pictures, torch.tensor([identity_label for i in range(len(pictures))])

    # [old] get item for the smaller easier sets (8 pictures per id)
    # def __getitem__(self, idx):
    #     pictures = []
    #     identity_path, identity_label = self.identity_paths_labels[idx]
    #     for img in os.listdir(identity_path):
    #         image_path = os.path.join(identity_path, img)
    #         tensor_image = image_path_to_tensor(image_path)
    #         # if self.transform:
    #         #     tensor_image = self.transform(tensor_image)
    #         if tensor_image is None:
    #             continue
    #         pictures.append(tensor_image)
    #
    #     if len(pictures) != 8:
    #         print("BAD len:", len(pictures), "identity: ", identity_label)
    #         for i in range(8 - len(pictures)):  # Cheap fix to make total 8 pictures
    #             pictures.append(pictures[i])
    #     # print(pictures)
    #     # print("len", len(pictures))
    #     pictures = torch.stack(pictures)
    #     return pictures, torch.tensor([identity_label for i in range(len(pictures))])



class ResNetEmbedding(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super(ResNetEmbedding, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features,
                                       embedding_dim)

    def forward(self, x):
        return self.base_model(x)


# class MySimpleResNet(nn.Module):
#     def __init__(self, embedding_size=128, keep_prob=0.5):
#         super(MySimpleResNet, self).__init__()

#         # Define layers
#         self.keep_prob = keep_prob

#         # Define ResNet blocks
#         self.resnet_block1 = self.resnet_block(3, 16, 3)
#         self.resnet_block2 = self.resnet_block(16, 32, 3)
#         self.resnet_block3 = self.resnet_block(32, 48, 3)
#         self.resnet_block4 = self.resnet_block(48, 64, 3)

#         # Define pooling layer
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Fully connected layer for embeddings
#         #self.fc1 = nn.Linear(64 * 4 * 4, embedding_size)  # Adjust dimensions as needed
#         #self.fc1 = nn.Linear(12544, embedding_size)
#         self.fc1 = nn.Linear(9152, embedding_size)

#     def resnet_block(self, in_channels, filters, k_size=3):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, filters, kernel_size=(k_size, k_size), padding='same',
#                       bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(filters, filters, kernel_size=(k_size, k_size), padding='same',
#                       bias=False),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         x = self.resnet_block1(x)
#         x = self.pool(x)

#         x = self.resnet_block2(x)
#         x = self.pool(x)

#         x = self.resnet_block3(x)
#         x = self.pool(x)

#         x = self.resnet_block4(x)
#         x = self.pool(x)

#         # Flatten
#         x = torch.flatten(x, 1)  # Flatten all dimensions except batch

#         # Dropout
#         x = F.dropout(x, p=1 - self.keep_prob, training=self.training)

#         # Embedding layer
#         embedding = self.fc1(x)  # Output size is embedding_size
#         return embedding



################################# TRAINING STARTS HERE #################################


if __name__ == "__main__":
    # torch.cuda.empty_cache()

    dataset = CustomDataset(root_dir=dataset_path, transform=transform,
                            size=TRAIN_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4)

    train_dataset = CustomDataset(root_dir=dataset_path, transform=transform,
                                  size=VAL_SIZE)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=4)

    val_dataset = CustomDataset(root_dir=val_dataset_path, transform=transform,
                                size=VAL_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=4)

    NUM_CLASSES = len(dataset.identity_paths_labels)

    # model = MySimpleResNet(embedding_size=128, keep_prob=0.5).to(device)
    # model = ResNetEmbedding(embedding_dim=EMBEDDING_DIM).to(device)
    model = MobileFaceNet(embedding_size=EMBEDDING_DIM).to(device)
    #model = torch.load("./models/cosinecustomCombinedMODEL512_batch16_total20000_epochs10_lr0.001_margin0.2_time2024-11-22@14-23-35.pth").to(device)

    # load pre-trained weights for MobileFaceNet
    load_model_path = './auxiliary_models/Epoch_17.pt'
    state_dict_loaded = model.state_dict()
    #state_dict_pretrained = torch.load(load_model_path, map_location=device)
    state_dict_pretrained = torch.load(load_model_path, map_location=device)['state_dict']
    state_dict_temp = {}

    # state_dict = torch.load(load_model_path)['state_dict']
    # model.load_state_dict(state_dict)

    for k in state_dict_loaded:
        if 'bn.num_running_batches' not in k:
            state_dict_temp[k] = state_dict_pretrained['backbone.'+k]
        else:
            print(k, 'not loaded!')
    state_dict_loaded.update(state_dict_temp)
    model.load_state_dict(state_dict_loaded)
    del state_dict_loaded, state_dict_pretrained, state_dict_temp




    # model = InceptionResnetV1(pretrained='vggface2', classify=False, device='cuda')
    # #pytorch-facenet FREEZE LAYERS:
    # # Freeze the backbone layers (early layers)
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # # Unfreeze the last block for fine-tuning
    # # Assuming InceptionResnetV1 has a 'last_linear' and 'logits' layer for the final layers
    # # This may vary depending on the model's structure
    #
    # # Unfreeze the last fully connected layers (logits)
    # for param in model.last_linear.parameters():
    #     param.requires_grad = True

    START_EPOCH = 0

    print("MODEL: ")
    # (model.base_model)
    # print(dir(model.base_model))

    print("----")

    # MobileFaceNets FREEZE LAYERS
    bn_moment = 0.9
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.momentum = bn_moment
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            bn_flag = 1
            if bn_flag == 0 or bn_flag == 1:
                layer.weight.requires_grad = True
                layer.bias.requires_grad = True

    for name, param in model.named_parameters():
        param.requires_grad = True
        # if name in ['linear.weight', 'linear.bias', 'bn.weight', 'bn.bias']:
        #     param.requires_grad = True
        # else:
        #     param.requires_grad = False


    # # ResNet FREEZE LAYERS
    # for param in model.base_model.parameters():
    #     param.requires_grad = False  # Freeze all layers
    #
    # # Unfreeze layer 4 and above
    # for param in model.base_model.layer4.parameters():
    #     param.requires_grad = True
    #
    # for param in model.base_model.fc.parameters():
    #     param.requires_grad = True


    #classifier = nn.Linear(EMBEDDING_DIM, NUM_CLASSES).to(device)
    #classifier = CosFaceLoss(NUM_CLASSES, EMBEDDING_DIM, margin=0.35, scale=64)
    classifier = ArcFace(EMBEDDING_DIM, NUM_CLASSES, margin_arc=0.35, margin_am=0.0, scale=32).to(device)

    for param in classifier.parameters():
        param.requires_grad = True


    parameters_model = [p for p in model.parameters() if p.requires_grad]
    parameters_sunglass_fc = [p for p in classifier.parameters() if
                              p.requires_grad]

    optimizer = optim.Adam([{'params': parameters_model},
                            {'params': parameters_sunglass_fc,
                             'lr': LEARNING_RATE, 'weight_decay': 1e-5}, ],
                           lr=LEARNING_RATE, weight_decay=1e-5)
    # Set up a scheduler to reduce the learning rate
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                  factor=0.1, patience=2,
    #                                                  threshold=0.2,
    #                                                  verbose=True)

    start_time = timer()
    train_losses = []
    train_losses_criterion = []
    val_losses = []
    val_losses_criterion = []
    lr_list = []
    eer_list = []

    for epoch in range(EPOCHS):
        model.train()  # Set the model to training mode
        classifier.train()

        epoch_start = timer()
        running_loss = 0.0
        batch_cnt = 0

        for identities, labels in dataloader:
            batch_cnt += 1
            identities_filtered = [id for id in identities if len(id) == IMGS_PER_ID]
            images = torch.stack(list(chain.from_iterable(identities_filtered)))
            labels_filtered = [lbl for lbl in labels if len(lbl) == IMGS_PER_ID]
            all_labels = torch.stack(list(chain.from_iterable(labels_filtered)))

            images, matching_labels = images.to(device), all_labels.to(device)

            # Forward pass to get embeddings
            embeddings = model(images)  # Output shape: (batch_size, EMBEDDING_DIM)

            # Add a classifier prediction
            predictions = classifier(embeddings, matching_labels)

            # Mine triplets
            semihard_triplets = miner(embeddings, matching_labels)

            # Calculate triplet loss
            if semihard_triplets is None or len(semihard_triplets) == 0:
                print("epoch ", epoch + 1, "no fitting triplets found")
                continue

            crit_loss = criterion(predictions, matching_labels)
            # TripletLoss:
            main_loss = 100*loss_func(embeddings, matching_labels, semihard_triplets)
            # ContrastiveLoss:
            #main_loss = loss_func(embeddings.unsqueeze(1), matching_labels)

            loss = main_loss + crit_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #curr_loss = loss.item()
            curr_loss = len(semihard_triplets[0]) * loss.item()
            curr_loss /= BATCH_TRIOS
            running_loss += curr_loss
            if batch_cnt % 10 == 0:
                print("epoch ", epoch + 1, f" batch {batch_cnt}, fitting triplets found: ", len(semihard_triplets[0]), " out of ", BATCH_TRIOS)
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {curr_loss}, MainLoss: {main_loss.item()}, CriterionLoss: {crit_loss.item()}")
        epoch_end = timer()
        # train_losses.append(running_loss/batch_cnt)
        print(
            f"Epoch {epoch + 1}, Time (seconds): {epoch_end - epoch_start}, Average loss: {running_loss / batch_cnt}")
        print("computing loss on train and validation sets:")

        model.eval()  # Set the model to evaluation mode (important for inference)
        classifier.eval()
        with torch.no_grad():  # Evaluations on TRAIN set
            model_true_actual_false = 0
            model_false_actual_true = 0
            model_false_actual_false = 0
            model_true_actual_true = 0
            duos_counter = 0
            same_duos_counter = 0
            diff_duos_counter = 0

            m_distances = []
            matches = []  # 1 for "same person" 0 for "diff person"

            val_running_loss = 0.0
            val_running_loss_criterion = 0.0
            val_batch_cnt = 0

            for identities, labels in train_dataloader:  # calculate train loss
                val_batch_cnt += 1
                identities_filtered = [id for id in identities if
                                       len(id) == IMGS_PER_ID]
                images = torch.stack(
                    list(chain.from_iterable(identities_filtered)))
                labels_filtered = [lbl for lbl in labels if
                                   len(lbl) == IMGS_PER_ID]
                all_labels = torch.stack(
                    list(chain.from_iterable(labels_filtered)))

                images, matching_labels = images.to(device), all_labels.to(
                    device)

                # Forward pass to get embeddings
                embeddings = model(images)  # Output shape: (batch_size, EMBEDDING_DIM)

                predictions = classifier(embeddings, matching_labels)

                # Mine triplets using semi-hard negative mining
                semihard_triplets = miner(embeddings, matching_labels)

                # Calculate triplet loss
                if semihard_triplets is None or len(semihard_triplets) == 0:
                    print("epoch ", epoch + 1, "no fitting triplets found")
                    continue

                # print(semihard_triplets)
                # for anchor, positive, negative in semihard_triplets:
                #TripletLoss:
                loss = 100*loss_func(embeddings, matching_labels, semihard_triplets)

                # ContrastiveLoss:
                #loss = loss_func(embeddings.unsqueeze(1), matching_labels)
                loss_criterion = criterion(predictions, matching_labels)
                curr_loss_criterion = loss_criterion.item()
                curr_loss = len(semihard_triplets[0]) * loss.item()
                curr_loss /= BATCH_TRIOS
                #curr_loss = loss.item()
                val_running_loss += curr_loss
                val_running_loss_criterion += curr_loss_criterion

            # Log results
            train_losses.append(val_running_loss / val_batch_cnt)
            train_losses_criterion.append(val_running_loss_criterion / val_batch_cnt)
            print("EVALUATION: Train average loss ",val_running_loss / val_batch_cnt)
            print("EVALUATION: Train average loss_criterion ",val_running_loss_criterion / val_batch_cnt)
            # scheduler.step(val_running_loss / val_batch_cnt)
            # lr_list.append(*(scheduler.get_last_lr()))

        model.eval()  # Set the model to evaluation mode (important for inference)
        classifier.eval()
        with torch.no_grad():  # Evaluations on Eval or Test set
            model_true_actual_false = 0
            model_false_actual_true = 0
            model_false_actual_false = 0
            model_true_actual_true = 0
            duos_counter = 0
            same_duos_counter = 0
            diff_duos_counter = 0

            m_distances = []
            matches = []  # 0 for "same person" 1 for "diff person"

            val_running_loss = 0.0
            val_running_loss_criterion = 0.0
            val_batch_cnt = 0

            for identities, labels in val_dataloader:  # calculate eval / test loss
                val_batch_cnt += 1
                identities_filtered = [id for id in identities if
                                       len(id) == IMGS_PER_ID]
                images = torch.stack(
                    list(chain.from_iterable(identities_filtered)))
                labels_filtered = [lbl for lbl in labels if
                                   len(lbl) == IMGS_PER_ID]
                all_labels = torch.stack(
                    list(chain.from_iterable(labels_filtered)))

                images, matching_labels = images.to(device), all_labels.to(
                    device)

                # Forward pass to get embeddings
                embeddings = model(images)  # Output shape: (batch_size, EMBEDDING_DIM)

                # Mine triplets using semi-hard negative mining
                semihard_triplets = miner(embeddings, matching_labels)

                # Calculate triplet loss
                if semihard_triplets is None or len(semihard_triplets) == 0:
                    print("epoch ", epoch + 1, "no fitting triplets found")
                    continue

                # print(semihard_triplets)
                # for anchor, positive, negative in semihard_triplets:
                #TripletLoss:
                loss = 100*loss_func(embeddings, matching_labels,semihard_triplets)
                # ContrastiveLoss:
                #loss = loss_func(embeddings.unsqueeze(1), matching_labels)
                loss_criterion = None
                curr_loss_criterion = 1  ## ## temporary fix
                curr_loss = len(semihard_triplets[0]) * loss.item()
                curr_loss /= BATCH_TRIOS
                #curr_loss = loss.item()
                val_running_loss += curr_loss
                val_running_loss_criterion += curr_loss_criterion

                #if epoch + 1 == EPOCHS:
                if True:
                    # Log distances for histograms
                    # Generate all unique duos (2-tuples) of images and corresponding labels
                    duos = itertools.combinations(range(len(embeddings)), 2)
                    # print("DUOS", duos)
                    # Iterate over the trios and get the corresponding images and labels
                    for idx1, idx2 in duos:
                        # if idx1 % 100 == 0 and idx2 % 100 == 0:
                        #     print("idx1, idx2 ", idx1, idx2)
                        duos_counter += 1
                        img1_l = matching_labels[idx1]
                        img2_l = matching_labels[idx2]
                        # dist = F.pairwise_distance(embeddings[idx1], embeddings[idx2])  # hope this does euclid dist
                        dist = DISTANCE_FUNC(embeddings[idx1].unsqueeze(0),
                                             embeddings[idx2].unsqueeze(0))
                        dist = dist.item()
                        m_distances.append(dist)
                        matches.append(int(img1_l == img2_l))

            # Log results
            val_losses.append(val_running_loss / val_batch_cnt)
            val_losses_criterion.append(val_running_loss_criterion / val_batch_cnt)
            print("EVALUATION: Val average loss ", val_running_loss / val_batch_cnt)
            #scheduler.step(val_running_loss / val_batch_cnt)
            scheduler.step()
            lr_list.append(scheduler.get_last_lr()[0])

            if epoch + 1 == 10 or epoch + 1 == 20 or epoch + 1 == 30:  # save checkpoints
                model_name = f'{DISTANCE_FUNC_NAME}customCombinedMODEL{EMBEDDING_DIM}_batch{BATCH_SIZE}_total{TRAIN_SIZE}_epochs{START_EPOCH + epoch + 1}_lr{LEARNING_RATE}_margin{MARGIN}_time{time.strftime("%Y-%m-%d@%H-%M-%S", time.localtime())}.pth'
                torch.save(model, os.path.join("./models", model_name))

            # perform accuracy tests bcz we already have data for it...
            m_distances = np.array(m_distances, dtype='float32')
            matches = np.array(matches, dtype='int32')
            # print("dists:", distances[:20])
            # print("matches", matches[:20])
            p_same_data = m_distances[matches == 1]
            p_diff_data = m_distances[matches == 0]
            ROC_labels = matches
            ROC_data = m_distances

            if epoch + 1 == EPOCHS:  # last epoch, save graph
                plt.figure(figsize=(10, 6))

                sns.histplot(p_same_data, bins=50, kde=True, color='green',
                             stat='density')  # kde=True adds the KDE curve
                sns.histplot(p_diff_data, bins=50, kde=True, color='red',
                             stat='density')  # kde=True adds the KDE curve

                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.title("Empirical Distribution (Histogram with KDE)")
                plt.savefig(
                    f'./results/Histograms_time{time.strftime("%Y-%m-%d@%H-%M-%S", time.localtime())}.png')
                # plt.show()
                # Compute the ROC curve

            fpr, tpr, thresholds = roc_curve(ROC_labels, ROC_data)
            # print(fpr)
            # print(tpr)

            eer, min_index = compute_eer(fpr, tpr)  # closer to 0 is better! 50% is worst
            #if DISTANCE_FUNC_NAME == 'cosine':
            # Assuming fpr and tpr are numpy arrays or lists
            optimal_threshold = thresholds[min_index]
            # # Calculate Youden's J statistic for each threshold
            # j_scores = tpr - fpr
            #
            # # Find the optimal threshold (highest J statistic)
            # optimal_index = np.argmax(j_scores)
            # optimal_threshold = thresholds[optimal_index]

            print("Optimal Threshold:", optimal_threshold)
            if DISTANCE_FUNC_NAME == 'euc':
                model_true_actual_false = len(
                    p_diff_data[p_diff_data < optimal_threshold])
                model_false_actual_true = len(
                    p_same_data[p_same_data >= optimal_threshold])
                model_false_actual_false = len(
                    p_diff_data[p_diff_data >= optimal_threshold])
                model_true_actual_true = len(
                    p_same_data[p_same_data < optimal_threshold])
            if DISTANCE_FUNC_NAME == 'cosine':
                model_true_actual_false = len(
                    p_diff_data[p_diff_data >= optimal_threshold])
                model_false_actual_true = len(
                    p_same_data[p_same_data < optimal_threshold])
                model_false_actual_false = len(
                    p_diff_data[p_diff_data < optimal_threshold])
                model_true_actual_true = len(
                    p_same_data[p_same_data >= optimal_threshold])

            model_correct = model_true_actual_true + model_false_actual_false
            true_catch = model_true_actual_true / (
                        model_true_actual_true + model_false_actual_true)
            false_catch = model_false_actual_false / (
                        model_false_actual_false + model_true_actual_false)
            eer_list.append(eer)
            print("True catch rate: ", true_catch)
            print("False catch rate: ", false_catch)
            print("Total accuracy [correct/total]: ",model_correct / duos_counter)
            print("EER ", eer)
            print(f"End of epoch {epoch+1} evaluation.")

    end_time = timer()

    model_name = f'{DISTANCE_FUNC_NAME}customCombinedMODEL{EMBEDDING_DIM}_batch{BATCH_SIZE}_total{TRAIN_SIZE}_epochs{START_EPOCH + EPOCHS}_lr{LEARNING_RATE}_margin{MARGIN}_time{time.strftime("%Y-%m-%d@%H-%M-%S", time.localtime())}.pth'
    torch.save(model, os.path.join("./models", model_name))

    print("Testing hyper params completed. Time taken (seconds): ",
          end_time - start_time)
    print("Val Losses list by epoch: ", val_losses)
    print("Train Losses list by epoch: ", train_losses)
    print("Learning Rate by epoch: ", lr_list)
    print("EER by epoch: ", eer_list)
    # Write the lr list to a text file
    lr_list_file_path = f'./results/LRlist_time{time.strftime("%Y-%m-%d@%H-%M-%S", time.localtime())}.txt'
    with open(lr_list_file_path, "w") as file:
        for item in lr_list:
            file.write(f"{item}\n")

    plt.figure(figsize=(10, 6))

    x_values = [x for x in range(1, EPOCHS + 1)]
    plt.plot(x_values, val_losses, label='Val loss', color='blue')
    plt.plot(x_values, train_losses, label="Train loss", color='red')

    plt.xlabel("EPOCH")
    plt.ylabel("Val Loss")
    plt.title("Validation loss per epoch")
    plt.legend()
    plt.savefig(
        f'./results/Losses_time{time.strftime("%Y-%m-%d@%H-%M-%S", time.localtime())}.png')
    # plt.show()

    # another graph for classification loss
    plt.figure(figsize=(10, 6))

    x_values = [x for x in range(1, EPOCHS + 1)]
    plt.plot(x_values, val_losses_criterion, label='Val loss_critirion', color='blue')
    plt.plot(x_values, train_losses_criterion, label="Train loss_critirion", color='red')

    plt.xlabel("EPOCH")
    plt.ylabel("Val Loss")
    plt.title("Validation loss_critirion per epoch")
    plt.legend()
    plt.savefig(
        f'./results/Losses_critirion_time{time.strftime("%Y-%m-%d@%H-%M-%S", time.localtime())}.png')


