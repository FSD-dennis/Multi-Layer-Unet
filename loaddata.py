from pacKage import *
from uNet import *

__all__ = ["minmax","TrainDataset", "TestDataset","train_batch", "accuracy"]


def minmax(image):
    """Scale the image data to [0, 255]."""
    min_val = np.min(image)
    max_val = np.max(image)
    # Avoid division by zero
    if max_val - min_val != 0:
        image = (image - min_val) / (max_val - min_val) * 255
    else:
        image = np.zeros_like(image)  # Optional: handle the case where max == min by setting to all zeros
    return image

class TrainDataset(Dataset):
    """_summary_
        Args:
            train_dir (Path):  "train image folder"
            label_dir (Path):  "label image folder"
            type (str, optional):  Defaults to "train".
            val_size (float, optional):  Defaults to 0.2.
            sep (str, optional):  Defaults to "ince". and another to "rgb"
    """
    def __init__(self, train_dir : Path, label_dir : Path, type : str = "train", val_size : float = 0.2, sep : str = "ince") -> None:
        self.sep = sep
        if self.sep not in ["ince", "rgb"]:
            raise(ValueError("only ince and rgb here"))
        
        self.fpaths = glob(train_dir + "/*")
        self.lpaths = glob(label_dir +"/*")
        
        seed(42)
        shuffle_path = list(zip(self.fpaths, self.lpaths))
        shuffle(shuffle_path)
        self.fpaths, self.lpaths = zip(*shuffle_path)
        
        index_len = int(len(self.fpaths) * (1 - val_size))
        
        if type not in ["train", "val"]:
            raise(ValueError("only, train and val is allowed"))
        elif type == "train":
            self.fpaths = self.fpaths[:index_len]
            self.lpaths = self.lpaths[:index_len]
        elif  type == "val":
            self.fpaths = self.fpaths[index_len:]
            self.lpaths = self.lpaths[index_len:]
            
        
    def __len__(self):
        return len(self.fpaths)
    
    def __getitem__(self, index) -> Any:
        with tiff.TiffFile(self.fpaths[index]) as train_temp:
            train_temp = train_temp.asarray()
        with tiff.TiffFile(self.lpaths[index]) as label_temp:
            label_temp = label_temp.asarray()
        
        if self.sep == "ince":
            temp_index = [0,1,5,6]
            train_temp = minmax(train_temp[:,:,temp_index])
            return torch.Tensor(train_temp).permute(2,0,1).to(device=device) ,torch.Tensor(label_temp).long().to(device=device)
        else:
            train_temp = minmax(train_temp[:,:,2:5])
            return torch.Tensor(train_temp).permute(2,0,1).to(device=device) ,torch.Tensor(label_temp).long().to(device=device)
        
        
class TestDataset(Dataset):
    def __init__(self, test_dir : Path) -> None:
        """_summary_
        Args:
            test_dir (Path) : "test image folder"
        """
        self.tpaths = glob(test_dir + "/*")
                
    def __len__(self):
        return len(self.tpaths)
    
    def __getitem__(self, index) -> Any:
        with tiff.TiffFile(self.tpaths[index]) as test_temp:
            test_temp = test_temp.asarray()
        name = self.tpaths[index].split("\\")[-1]
        
        temp_index = [0,1,5,6]
        ince = test_temp[:,:,temp_index]
        rgb = test_temp[:,:,2:5]
        return torch.Tensor(ince).permute(2,0,1).to(device=device), torch.Tensor(rgb).permute(2,0,1).to(device=device), name
    
    
def train_batch(x, y, model, loss_func, optimizer):
    prediction = model(x)
    loss_fn = loss_func
    optimizer = optimizer
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x).argmax(1)
    accuracy = (prediction == y)
    sample = prediction.numel()
    accuracy = accuracy.sum()/sample
    return accuracy.detach().cpu().numpy()



# class auto_dataset():
#     def __init__(self, folder_train: Path, folder_target: Path, split_type="train", val_size=0.2) -> None:
#         self.split_type = split_type
#         # Populate file paths
#         self.fpaths = glob(folder_train + "/*")
#         self.tpaths = glob(folder_target + "/*")
    
#         # Calculate split index based on val_size
#         split_index = int(len(self.fpaths) * (1 - val_size))
    
#         # Create indices for train and val sets
#         indices = list(range(len(self.fpaths)))
#         if split_type == "train":
#             self.train_indices = indices[:split_index]
#         elif split_type == "val":
#             self.val_indices = indices[split_index:]
#         elif split_type == "all":
#             self.train_indices = indices
#             self.val_indices = list(range(len(self.tpaths)))
        
#         # Initialize MinMaxScaler
#         self.scaler = MinMaxScaler()

    
    
    
    # def __len__(self) -> int:
    #     if self.split_type == "train":
    #         return len(self.train_indices)
    #     elif self.split_type == "val":
    #         return len(self.val_indices)
    #     elif self.split_type == "all":
    #         return len(self.train_indices) + len(self.val_indices)
    
    # def __getitem__(self, index) -> Tuple[torch.Tensor]:
    #     with tiff.TiffFile(self.fpaths[index]) as train_temp:
    #         train = train_temp.asarray()
    #         # Normalize train data using MinMaxScaler
    #         train = self.scaler.fit_transform(train.reshape(-1, 1)).reshape(train.shape)

    #     with tiff.TiffFile(self.tpaths[index]) as label_temp:
    #         label = label_temp.asarray()
    #         # Normalize label data using MinMaxScaler
    #         label = self.scaler.fit_transform(label.reshape(-1, 1)).reshape(label.shape)
            
    #     return torch.Tensor(train).permute(2, 0, 1).to(device=device), torch.Tensor(label).to(device=device)


# class rgb_dataset(Dataset):
#     def __init__(self, folder_train: Path, folder_target: Path, split_type="train", val_size=0.2) -> None:
#         self.split_type = split_type
#         # Populate file paths
#         self.fpaths = glob(str(folder_train) + "/*")
#         self.tpaths = glob(str(folder_target) + "/*")
    
#         # Calculate split index based on val_size
#         split_index = int(len(self.fpaths) * (1 - val_size))
    
#         # Create indices for train and val sets
#         indices = list(range(len(self.fpaths)))
#         if split_type == "train":
#             self.train_indices = indices[:split_index]
#         elif split_type == "val":
#             self.val_indices = indices[split_index:]
#         elif split_type == "all":
#             self.train_indices = indices
#             self.val_indices = list(range(len(self.tpaths)))
        
#         # Initialize MinMaxScaler
#         self.scaler = MinMaxScaler()

#     def __len__(self) -> int:
#         if self.split_type == "train":
#             return len(self.train_indices)
#         elif self.split_type == "val":
#             return len(self.val_indices)
#         elif self.split_type == "all":
#             return len(self.train_indices) + len(self.val_indices)
    
#     def __getitem__(self, index) -> Tuple[torch.Tensor]:
#         with tiff.TiffFile(self.fpaths[index]) as train_temp:
#             train = train_temp.asarray()[:, :, 2:5]  # Take only the 3rd, 4th, and 5th layers
#             # Normalize train data using MinMaxScaler
#             train = self.scaler.fit_transform(train.reshape(-1, 1)).reshape(train.shape)

#         with tiff.TiffFile(self.tpaths[index]) as label_temp:
#             label = label_temp.asarray()[:, :, 2:5]  # Take only the 3rd, 4th, and 5th layers
#             # Normalize label data using MinMaxScaler
#             label = self.scaler.fit_transform(label.reshape(-1, 1)).reshape(label.shape)
            
#         return torch.Tensor(train).permute(2, 0, 1), torch.Tensor(label)


# class UNetWork(nn.Module):
#     def __init__(self, in_channels, out_channels, features):
#         self.ups = nn.ModuleList()
#         self.downs = nn.ModuleList()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Down part
#         for feature in features:
#             self.downs.append(DoubleConv(in_channels, feature))
#             in_channels = feature

#         # UP sampling part
#         for feature in reversed(features):
#             self.ups.append(
#                 nn.ConvTranspose2d(
#                     feature*2, feature, kernel_size=2, stride=2,
#                     )
#             )
#             self.ups.append(DoubleConv(feature*2, feature))
#         self.bottleneck = DoubleConv(features[-1], features[-1]*2)
#         self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

#     def forward(self, x):
#         print("Input size:", x.size())

#         skip_connections = []
#         for idx, down in enumerate(self.downs):
#             x = down(x)
#             skip_connections.append(x)
#             x = self.pool(x)
#             print(f"Down {idx + 1} output size:", x.size())

#         x = self.bottleneck(x)
#         skip_connections = skip_connections[::-1]

#         for idx in range(0, len(self.ups), 2):
#             x = self.ups[idx](x)
#             skip_connection = skip_connections[idx//2]
#             if x.shape != skip_connection.shape:
#                 x = F.resize(x, skip_connection.shape[2:])
#             concat_skip = torch.cat((skip_connection, x), dim=1)
#             x = self.ups[idx+1](concat_skip)
#             print(f"Up {idx // 2 + 1} output size:", x.size())

#         output = self.final_conv(x)
#         output = torch.sigmoid(output)
#         print("Final output size:", output.size())

#         return output