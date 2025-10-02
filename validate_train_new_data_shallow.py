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
import xarray as xr
import os
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as scio
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

#-----------------------------------------------HyperParameter-----------------------------------------------#

P = 13

z_dim = 20


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



bands = wavelengths.shape[0]

#--------------------------------------------------Model-------------------------------------------------------#
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
    
class MLP(nn.Module):
    def __init__(self, P, bands):
        super(MLP, self).__init__()
        self.P = P
        self.bands = bands
        self.fc1 = nn.Linear(bands, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)

        self.fc4 = nn.Linear(16, P)

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
        a = torch.abs(h4)
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

        alpha = torch.sigmoid(self.alpha)
        output =  alpha * linear_combination + (1-alpha) * non_linear_output
        return output

#------------------------------------------------------Load Rrs-----------------------------------------------------#   

file_ground_truth = "./Data/labeled_data/clean_Data/new/clean_rrs_699nm_all_new.csv"
df = pd.read_csv(file_ground_truth, header= None, dtype=float)
labeled_rrs = df.values.T 
labeled_rrs = labeled_rrs[1:, :]
min_values_gt = np.min(labeled_rrs, axis=1, keepdims=True)  # Find the minimum for each curve
max_values_gt = np.max(labeled_rrs, axis=1, keepdims=True)  # Find the maximum for each curve
normalized_rrs_gt = (labeled_rrs - min_values_gt) / (max_values_gt - min_values_gt + 1e-8) 
#normalized_rrs_gt = normalized_rrs_gt[:,selected_indices] # modified
Rrs = torch.tensor(normalized_rrs_gt, dtype=torch.float32)
print("RRS shape",Rrs.shape)

#----------------------------------------------Load Endmembers Ground Truth-----------------------------------------#

EM = pd.read_csv('./endmember/Complete_EM/Groundtruth_EM_new_699nm.csv', header=None, skiprows=1).iloc[:, 1:].to_numpy()
EM = EM.T
print(f"EM shape: {EM.shape}")  # (11, 159) num, wavelength
# Normalize each curve in EM to [0, 1]
min_values_em = np.min(EM, axis=1, keepdims=True)  # Find the minimum for each curve
max_values_em = np.max(EM, axis=1, keepdims=True)  # Find the maximum for each curve
EM_normalized = (EM - min_values_em) / (max_values_em - min_values_em + 1e-8)
#EM_normalized = EM_normalized[:, selected_indices]
EM = torch.tensor(EM_normalized, dtype=torch.float32).to(device)
print(f"EM Tensor shape: {EM.shape}")

#----------------------------------------------Load Endmembers Ground Truth-----------------------------------------#

gt_abundances = pd.read_csv('./Data/labeled_data/clean_Data/new/clean_gt_all_new.csv', header=None, skiprows=1).iloc[:, 1:].values
gt_abundances = gt_abundances / 100.0
gt_abundances = torch.tensor(gt_abundances, dtype=torch.float32).to(device)
print("GT Abundance shape:", gt_abundances.shape)

#---------------------------------------------------Other Function------------------------------------------------------#
def EM_with_weight(a, EM):
    a_expanded = a.unsqueeze(-1)
    EM_expanded = EM.unsqueeze(0)
    output = a_expanded * EM_expanded
    return output

def criteria_penalty(Rrs_input, abundance, wavelengths, alpha=1.0):
    """
    Add penalty if 620nm ≈ 650nm and abundance of certain types is high.
    """
    idx_620 = np.argmin(np.abs(wavelengths - 620))
    idx_650 = np.argmin(np.abs(wavelengths - 650))

    rrs_620 = Rrs_input[:, idx_620]  # shape: [B]
    rrs_650 = Rrs_input[:, idx_650]  # shape: [B]


    mask = torch.abs(rrs_650 - rrs_620) < 0.002  
    a_sub = abundance[:, [0, 1, 2]]  # shape: [B, 3]
    a_sum = torch.sum(a_sub, dim=1)  # shape: [B]

    penalty = torch.where(mask, a_sum, torch.zeros_like(a_sum))

    return alpha * penalty.mean()
#---------------------------------------------------dir------------------------------------------------------#

output_dir = "./Output/all_new/test40"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#---------------------------------------------------Train Test Split------------------------------------------------------#
num_rounds = 15
test_size = 40

num_samples = len(Rrs)
indices = list(range(num_samples))

top1_correct = 0
top2_correct = 0
top3_correct = 0
top4_correct = 0
total_samples = 0
all_abundance_records = []

idx_650 = np.where(wavelengths == 650)[0][0]
idx_620 = np.where(wavelengths == 620)[0][0]

for round_idx in range(num_rounds):
    print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")

    torch.manual_seed(42 + round_idx)
    np.random.seed(42 + round_idx)
    torch.cuda.manual_seed(42 + round_idx)
    random.seed(42 + round_idx)
    test_indices = random.sample(indices, test_size)
    train_indices = [i for i in indices if i not in test_indices]

    train_data = Rrs[train_indices]
    test_data = Rrs[test_indices]
    train_gt = gt_abundances[train_indices]
    test_gt = gt_abundances[test_indices]

    train_gt_tensor = torch.tensor(train_gt, dtype=torch.float32)
    test_gt_tensor = torch.tensor(test_gt, dtype=torch.float32)

    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)

    train_dataset = TensorDataset(train_tensor, train_gt_tensor)
    test_dataset = TensorDataset(test_tensor, test_gt_tensor)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    train_set = set(tuple(row.tolist()) for row in train_data)
    test_set = set(tuple(row.tolist()) for row in test_data)

#-------------------------------------------------- Train Model ---------------------------------------------------------#
    model = MLP(P, bands).to(device)
    decoder = Decoder(P, wavelength_length=bands).to(device)

    abun_weight = torch.nn.Parameter(torch.ones(6, device=device)) 
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(decoder.parameters()) + [abun_weight],
        lr=1e-4,
        weight_decay=1e-4
    )

    epochs = 70

    losses = []
    print('Start training!')
    noise_std=0.01

    for epoch in range(epochs):
        model.train()
        decoder.train()
        epoch_loss = 0
        epoch_loss_recon = 0.0
        abundances = []

        for step, (y, a_gt) in enumerate(train_loader):
            y = y.to(device)   # [1000, 159]
            a_gt = a_gt.to(device)
            a = model(y)          # [1000, 10]
            em_wt = EM_with_weight(a, EM)
            y_hat = decoder(em_wt)

            a0 = a[:, 0:1]
            a1 = a[:, 1:2] + a[:, 2:3]
            a2 = a[:, 3:4] + a[:, 4:5]
            a3 = a[:, 5:6] + a[:, 6:7]
            a4 = a[:, 7:8] + a[:, 8:9]
            a5 = a[:, 9:10] + a[:, 10:11]
            a_grouped = torch.cat([a0, a1, a2, a3, a4, a5], dim=1)

            a_cdom = a[:, 11:12]
            a_nap = a[:, 12:13]
            #a_grouped_scaled = a_gt * (1 - a_cdom -a_nap)

            #cyano 620-650
            rrs_650 = y[:, idx_650]
            rrs_620 = y[:, idx_620]
            a_target = a_grouped[:, 2]
            condition_mask = (rrs_650 > rrs_620).float()
            inverse_mask = 1.0 - condition_mask
            #target_penalty = F.relu(0.2 - a_target) 
            rrs_diff = rrs_650 - rrs_620
            scale_factor = 2.5
            target_penalty = torch.clamp(rrs_diff * scale_factor, min=0.0, max=1.0)
            penalty_when_positive = condition_mask * target_penalty  
            #penalty_loss = torch.mean(condition_mask * target_penalty)

            a_diatom = a_grouped[:, 3]
            diatom_penalty = F.relu(0.4 - a_diatom)
            penalty_when_negative = inverse_mask * diatom_penalty

            penalty_loss = torch.mean(penalty_when_positive )

            # cyano high, dino hapto low
            a2 = a_grouped[:, 2]
            a4 = a_grouped[:, 4]
            a2_mask = (a2 > 0.25).float()
            mutual_penalty = torch.mean(a2_mask* a2 * (a4))

            mse = (a_grouped - a_gt) ** 2 
            max_indices = torch.argmax(a_gt, dim=1)
            batch_size = a_gt.shape[0]
            weights = torch.ones_like(mse) 
            dominant_weight_per_sample = abun_weight[max_indices]
            weights.scatter_(1, max_indices.unsqueeze(1), dominant_weight_per_sample.unsqueeze(1))
            loss_abun = torch.mean(weights * mse)

            loss_recon = torch.mean((y_hat - y) ** 2)
            #loss_abun = torch.mean((a_grouped - a_gt)**2) #modified
            loss = 0.1 * loss_recon + loss_abun + 0.1 * penalty_loss + 0.1*mutual_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_recon += loss_recon.item()

            a_np = a.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            abundances.append((a_np, y_np))

        losses.append(loss.detach().cpu().numpy())
        print(f"Epoch [{epoch + 1}/{epochs}] completed, Average Loss: {epoch_loss / len(train_loader):.8f}, Recon loss:{epoch_loss_recon / len(train_loader):.8f}")

        # Save abundances and model weights

        if epoch == epochs - 1:
            all_abundances_with_y = [(a_batch, y_batch) for a_batch, y_batch in abundances]
            abundance_file = os.path.join(output_dir, f"abundance_with_y_epoch_{epoch + 1}.npy")

            # Save using NumPy (saves the object directly)
            import pickle
            with open(abundance_file, 'wb') as f:
                pickle.dump(all_abundances_with_y, f)

            print(f"Saved abundances and ground truths with indices for last epoch to {abundance_file}")

            torch.save(model.state_dict(), os.path.join(output_dir, f"mlp_model_weights_{round_idx+1}.pth"))
            torch.save(decoder.state_dict(), os.path.join(output_dir, f"decoder_weights_{round_idx+1}.pth"))
            print(f"Saved model weights!")

    #-----------------------------------------------------test---------------------------------------------------#
    model.eval()
    decoder.eval()
    test_loss_mse = 0
    test_loss_mae = 0
    test_loss_sad = 0
    random_indices = None
    y_sample = None
    y_hat_sample = None
    a_sample = None

    sample_indices = []

    test_indices = []
    test_original = []
    test_reconstructed = []
    test_abundances = []
    test_gt_abundances = []

    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    top4_correct = 0
    total_samples = 0


    with torch.no_grad():
        for step, (y, a_gt) in enumerate(test_loader):
            y = y.to(device)
            a = model(y)
            em_wt = EM_with_weight(a, EM)
            y_hat = decoder(em_wt)

            mse = torch.mean((y_hat - y) ** 2)
            mae = torch.mean(torch.abs(y_hat - y))

            cosine_similarity = torch.sum(y_hat * y, dim=1) / (torch.norm(y_hat, dim=1) * torch.norm(y, dim=1) + 1e-6)
            sad = torch.mean(torch.acos(torch.clamp(cosine_similarity, -1.0, 1.0)))

            test_loss_mse += mse.item()
            test_loss_mae += mae.item()
            test_loss_sad += sad.item()

            y_numpy = y.cpu().numpy()
            y_hat_numpy = y_hat.cpu().numpy()
            a_numpy = a.cpu().numpy()
            a_gt_numpy = a_gt.cpu().numpy()

            gt = a_gt.squeeze(0).cpu().numpy()
            a0 = a[:, 0:1]
            a1 = a[:, 1:2] + a[:, 2:3]
            a2 = a[:, 3:4] + a[:, 4:5]
            a3 = a[:, 5:6] + a[:, 6:7]
            a4 = a[:, 7:8] + a[:, 8:9]
            a5 = a[:, 9:10] + a[:, 10:11]
            a_grouped = torch.cat([a0, a1, a2, a3, a4, a5], dim=1)

            pred = a_grouped.squeeze(0).cpu().numpy()

            gt_index = int(gt.argmax())
            topk_pred_indices = pred.argsort()[::-1] 

            if gt_index == topk_pred_indices[0]:
                top1_correct += 1
            if gt_index in topk_pred_indices[:2]:
                top2_correct += 1
            if gt_index in topk_pred_indices[:3]:
                top3_correct += 1
            if gt_index in topk_pred_indices[:4]:
                top4_correct += 1

            total_samples += 1

            if len(sample_indices) < 40:
                available_indices = list(range(y_numpy.shape[0]))
                random_indices = random.sample(available_indices, min(400 - len(sample_indices), len(available_indices)))
                sample_indices.extend([(step, idx) for idx in random_indices])

                for idx in random_indices:
                    test_original.append(y_numpy[idx])
                    test_reconstructed.append(y_hat_numpy[idx])
                    test_abundances.append(a_numpy[idx])
                    test_gt_abundances.append(a_gt_numpy[idx])

    num_test_batches = len(test_loader)
    print(f"Test Loss (MSE): {test_loss_mse / num_test_batches:.8f}")
    print(f"Test Loss (MAE): {test_loss_mae / num_test_batches:.8f}")
    print(f"Test Loss (SAD): {test_loss_sad / num_test_batches:.8f}")

    top1_acc = top1_correct / total_samples if total_samples > 0 else 0
    top2_acc = top2_correct / total_samples if total_samples > 0 else 0
    top3_acc = top3_correct / total_samples if total_samples > 0 else 0
    top4_acc = top4_correct / total_samples if total_samples > 0 else 0

    with open("topk_accuracy_log.txt", "a") as f:
        f.write(f"=== Round {round_idx + 1}/{num_rounds} ===\n")
        f.write(f"Top-1 Accuracy: {top1_acc:.4f}\n")
        f.write(f"Top-2 Accuracy: {top2_acc:.4f}\n")
        f.write(f"Top-3 Accuracy: {top3_acc:.4f}\n")
        f.write(f"Top-4 Accuracy: {top4_acc:.4f}\n")
        f.write("\n")


    reconstructed_rrs_list = []
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20', 12)

    # draw test sample 
    for i in range(len(test_original)):

        a_labels = ["Chlorophyte", "Cryptophyte","Cyanophyte", "Diatom", "Dinoflagellate","Haptophyte", "cdom", "nap"]
        #a_labels = ["CDOM", "Chlorophyte", "Cryptophyte","Cyanophyte", "Diatom", "Dinoflagellate","Haptophyte", "NAP", "Prochlorococcus", "Synecoccus", "water"]

        plt.figure(figsize=(12, 6))
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 16  
        })
        
        a_for_sample_1 = test_abundances[i]  
        a_test_0 = a_for_sample_1[0]
        a_test_1 = a_for_sample_1[1]+a_for_sample_1[2]
        a_test_2 = a_for_sample_1[3]+a_for_sample_1[4]
        a_test_3 = a_for_sample_1[5]+a_for_sample_1[6]
        a_test_4 = a_for_sample_1[7]+a_for_sample_1[8]
        a_test_5 = a_for_sample_1[9]+a_for_sample_1[10]
        a_test_6 = a_for_sample_1[11]
        a_test_7 = a_for_sample_1[12]
        a_for_sample = torch.tensor([a_test_0, a_test_1, a_test_2, a_test_3, a_test_4, a_test_5, a_test_6, a_test_7], dtype=torch.float32)

        a_gt = test_gt_abundances[i]
        a_text = "\n".join([
            f"{a_labels[j] if j < len(a_labels) else f'EM_{j}'}: "
            f"{a_for_sample[j]:.4f} | gt={a_gt[j]:.4f}" if j < len(a_gt)
            else f"{a_labels[j] if j < len(a_labels) else f'EM_{j}'}: {a_for_sample[j]:.4f} | gt=N/A"
            for j in range(len(a_for_sample))
        ]) # the number of endmember are not equal

        plt.plot(wavelengths, test_original[i], label='Original Reflectance (y)', marker='o', linestyle='-',  color = '#3f37c9')
        plt.plot(wavelengths, test_reconstructed[i], label='Reconstructed Reflectance (y_hat)', marker='x', linestyle='--', color = '#d90429')
        plt.title(f"Comparison of Original and Reconstructed Reflectance (Sample {i+1})")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance Value")
        plt.legend()
        plt.grid()

        plt.subplots_adjust(right=0.7)  # Shrink the plot to create space on the right
        
        # Add the a values as text outside the plot
        plt.gca().text(1.02, 0.5, a_text, transform=plt.gca().transAxes, fontsize=14, 
                        verticalalignment='center', horizontalalignment='left', 
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))

        save_path = os.path.join(output_dir, f"reflectance_sample_{round_idx+1}_{i+1}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved reflectance comparison plot for sample {i+1} to {save_path}")

        pred_values = a_for_sample
        #print(f"Sample {i + 1} predicted values: {pred_values.numpy()}")
        scale = 1 - a_test_6 - a_test_7
        pred_values_scaled = pred_values / scale
        record = {
            "round": round_idx + 1,
            "sample_id": i + 1,
            "pred": pred_values_scaled,
            "gt": a_gt
        }
        all_abundance_records.append(record)


all_txt_save_path = os.path.join(output_dir, "all_abundances.txt")
with open(all_txt_save_path, 'w') as f:
    f.write("Round\tSample_ID\t" + "\t".join([f"Pred_{j}" for j in range(6)]) + "\t" + "\t".join([f"GT_{j}" for j in range(6)]) + "\n")
    for record in all_abundance_records:
        pred_str = "\t".join([f"{v:.6f}" for v in record["pred"][:6]])  
        gt_str = "\t".join([f"{v:.6f}" for v in record["gt"][:6]])
        f.write(f"{record['round']}\t{record['sample_id']}\t{pred_str}\t{gt_str}\n")
print(f"Saved all abundances across all rounds to {all_txt_save_path}")

print("\n=== Top-k Accuracy Summary ===")
print(f"Total samples: {total_samples:.4f}")
print(f"Top-1 Accuracy: {top1_correct / total_samples:.4f}")
print(f"Top-2 Accuracy: {top2_correct / total_samples:.4f}")
print(f"Top-3 Accuracy: {top3_correct / total_samples:.4f}")
print(f"Top-4 Accuracy: {top4_correct / total_samples:.4f}")