import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
import math
import earthaccess
import hypercoast
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as scio
import matplotlib.pyplot as plt
import random# 打开 NetCDF 文件
import hypercoast
import matplotlib as mpl
import rasterio
from rasterio.plot import show

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wavelengths = np.array([
    400., 403., 405., 408., 410., 413., 415., 418., 420., 422., 425., 427., 430., 432., 435., 437., 440., 442.,
    445., 447., 450., 452., 455., 457., 460., 462., 465., 467., 470., 472., 475., 477., 480., 482., 485., 487.,
    490., 492., 495., 497., 500., 502., 505., 507., 510., 512., 515., 517., 520., 522., 525., 527., 530., 532.,
    535., 537., 540., 542., 545., 547., 550., 553., 555., 558., 560., 563., 565., 568., 570., 573., 575., 578.,
    580., 583., 586., 588., 613., 615., 618., 620., 623., 625., 627., 630., 632., 635., 637., 640., 641., 642.,
    643., 645., 646., 647., 648., 650., 651., 652., 653., 655., 656., 657., 658., 660., 661., 662., 663., 665.,
    666., 667., 668., 670., 671., 672., 673., 675., 676., 677., 678., 679., 681., 682., 683., 684., 686., 687.,
    688., 689., 691., 692., 693., 694., 696., 697., 698., 699.
])

P = 13
z_dim = 20
bands = wavelengths.shape[0]

class MLP(nn.Module):
    def __init__(self, P, bands):
        super(MLP, self).__init__()
        self.P = P
        self.bands = bands
        self.fc1 = nn.Linear(bands, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        self.fc4 = nn.Linear(64, 16)
        self.bn4 = nn.BatchNorm1d(16)

        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, P)

    def gen_a(self, x):
        h1 = self.fc1(x)
        h1 = self.bn1(h1)
        h1 = F.leaky_relu(h1)

        h2 = self.fc2(h1)
        h2 = self.bn2(h2)
        h2 = F.leaky_relu(h2)

        h3 = self.fc3(h2)
        h3 = self.bn3(h3)
        h3 = F.leaky_relu(h3)

        h4 = self.fc4(h3)
        h4 = self.bn4(h4)

        h5 = self.fc5(h4)
        h6 = self.fc6(h5)

        a = torch.abs(h6)
        a = a / (a.sum(dim=1, keepdim=True) + 1e-8)
        #a = F.softmax(h6, dim = 1)
        return a
    
    def forward(self, inputs):
        a = self.gen_a(inputs)
        return a
    
class Decoder(nn.Module):
    def __init__(self, n, wavelength_length):
        super(Decoder, self).__init__()
        self.n = n
        self.wavelength_length = wavelength_length

        self.linear_weights = nn.Parameter(torch.randn(n, 1))  # [n, 1]
        
        self.non_linear = nn.Sequential(
            nn.Linear(n * wavelength_length, 256),
            nn.ReLU(),
            nn.Linear(256, wavelength_length),
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        batch_size = x.size(0)
        linear_combination = x.sum(dim=1) 
        x_flattened = x.reshape(batch_size, -1)  # [batch_size, n * wavelength_length]
        non_linear_output = self.non_linear(x_flattened)  # [batch_size, wavelength_length]
        output =  linear_combination + non_linear_output
        return output
    
# 替换为你的真实文件路径

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


model = MLP(P, bands).to(device)
decoder = Decoder(P, wavelength_length=bands).to(device)

round = [1,5,6,12]
for j in round:
    checkpoint_path_model = f"./Output/all_new/test40_deeper/mlp_model_weights_{j}.pth"
    if os.path.exists(checkpoint_path_model):
        checkpoint = torch.load(checkpoint_path_model, map_location=device)
        model.load_state_dict(checkpoint)  
        print("Loaded pretrained model weights for fine-tuning.")
    else:
        print("No pretrained model found, training model from scratch.")             

    save_dir = f"./images/time_series_all_new/test40_deeper/model{j}_abundance/"                                                                                                         
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    decoder.eval()


    folder_path = "./Data/PACE-001/Time_series/"
    nc_files = [f for f in os.listdir(folder_path) if f.endswith(".nc")]

    for filename in nc_files:
        file_id = os.path.splitext(filename)[0]
        file_path = os.path.join(folder_path, filename)
        print(f"Processing {file_id}...")

        ds = hypercoast.read_pace(file_path)
        ds_selected = ds["Rrs"].sel(wavelength=wavelengths, method="nearest")
        rrs_full = ds_selected.values
        print(rrs_full.shape)
        idx_400 = np.argmin(np.abs(wavelengths - 400))
        idx_500 = np.argmin(np.abs(wavelengths - 500))
        valid_mask = ~np.isnan(rrs_full).any(axis=2)
        rrs_400 = rrs_full[:, :, idx_400]
        rrs_500 = rrs_full[:, :, idx_500]
        value_mask = rrs_500 > rrs_400

        lat = ds["latitude"].values
        lon = ds["longitude"].values

        rgb_image_tif_file = "./tif/1024.tif"
        if os.path.exists(rgb_image_tif_file):
            with rasterio.open(rgb_image_tif_file) as src:
                bounds = src.bounds

        lat_mask = (lat >= bounds.bottom) & (lat <= bounds.top)
        lon_mask = (lon >= bounds.left) & (lon <= bounds.right)
        geo_mask = lat_mask & lon_mask

        combined_mask = valid_mask & value_mask & geo_mask
        lat_idx, lon_idx = np.where(combined_mask)
        rrs_valid = rrs_full[lat_idx, lon_idx, :]
        print(rrs_valid.shape)

        with torch.no_grad():
            rrs_tensor = torch.tensor(rrs_valid, dtype=torch.float32).to(device)
            min_values = torch.min(rrs_tensor, dim=1, keepdim=True).values
            max_values = torch.max(rrs_tensor, dim=1, keepdim=True).values
            normalized_rrs = (rrs_tensor - min_values) / (max_values - min_values + 1e-8)
            a = model(normalized_rrs)

            a0 = a[:, 0:1]
            a1 = a[:, 1:2] + a[:, 2:3]
            a2 = a[:, 3:4] + a[:, 4:5]
            a3 = a[:, 5:6] + a[:, 6:7]
            a4 = a[:, 7:8] + a[:, 8:9]
            a5 = a[:, 9:10] + a[:, 10:11]
            a_grouped = torch.cat([a0, a1, a2, a3, a4, a5], dim=1)
            a_pred = a_grouped.cpu().numpy()

        abundance_maps = [np.full(rrs_full.shape[:2], np.nan) for _ in range(6)]
        for k in range(6):
            abundance_maps[k][lat_idx, lon_idx] = a_pred[:, k]

        abundance_names = ["Chrolo", "Crypto", "Cyano", "Diatom", "Dino", "Hypto"]

        all_abundance_dict = {
        "Latitude": lat[lat_idx, lon_idx].flatten(),
        "Longitude": lon[lat_idx, lon_idx].flatten()
        }

        for k, name in enumerate(abundance_names):
            abundance_map = abundance_maps[k]
            vmin = 0.35
            vmax = 0.0

            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([bounds.left, bounds.right, bounds.bottom, bounds.top], crs=ccrs.PlateCarree())

            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.bottom_labels = True
            gl.left_labels = True
            gl.xlocator = plt.MaxNLocator(4)
            gl.ylocator = plt.MaxNLocator(4)

            gl.xlabel_style = {'size': 30, 'color': 'black'}  
            gl.ylabel_style = {'size': 30, 'color': 'black'} 

            # add rgb images
            rgb_image_tif_file = "./tif/1024.tif"
            if os.path.exists(rgb_image_tif_file):
                with rasterio.open(rgb_image_tif_file) as src:
                    bounds = src.bounds  
                    ax.set_extent([bounds.left, bounds.right, bounds.bottom, bounds.top], crs=ccrs.PlateCarree())
                    show(src, ax=ax, transform=src.transform)

            try:
                if np.isnan(vmin) or np.isnan(vmax):
                    raise ValueError("vmin or vmax is NaN")

                mesh = ax.pcolormesh(lon, lat, abundance_map, transform=ccrs.PlateCarree(),
                                    cmap="jet", rasterized=True, vmin=vmin, vmax=vmax)
                cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", label="Abundance")# shrink0.4
                cbar.set_ticks(np.linspace(vmin, vmax, 4))
                cbar.ax.tick_params(labelsize=30) 
                cbar.set_label("Abundance", fontsize=30)

            except Exception as e:
                print(f"[Warning] Cannot render abundance map for {name}: {e}")

            ax.coastlines(resolution="10m", linewidth=1)
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), linestyle=":")
            ax.add_feature(cfeature.LAND, facecolor="lightgray")

            output_path = os.path.join(save_dir, f"{file_id}_{name}_para.png")
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()

            Image.open(output_path).convert("RGB").save(output_path.replace(".png", ".pdf"), "PDF", resolution=150)
            print(f"Saved: {output_path}")

            valid_abundance = abundance_maps[k][lat_idx, lon_idx].flatten()
            all_abundance_dict[name] = valid_abundance

        df_all = pd.DataFrame(all_abundance_dict)
        csv_save_path = os.path.join(save_dir, f"{file_id}_all_abundance.csv")
        df_all.to_csv(csv_save_path, index=False)
        print(f"Saved CSV (all abundance): {csv_save_path}")
