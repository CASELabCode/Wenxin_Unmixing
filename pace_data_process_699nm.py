import os
import numpy as np
import hypercoast
import xarray as xr

# 设置路径
input_folder = "./Data/PACE-001/PACE"
output_folder = "./Data/PACE-001/Processed"
os.makedirs(output_folder, exist_ok=True)


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


idx_400 = np.argmin(np.abs(wavelengths - 400))
idx_500 = np.argmin(np.abs(wavelengths - 500))

all_rrs_list = []

for file in os.listdir(input_folder):
    if not file.endswith(".nc"):
        continue

    print(f"\nProcessing: {file}")
    file_path = os.path.join(input_folder, file)
    
    try:
        ds = hypercoast.read_pace(file_path)
    except Exception as e:
        print(f"Failed to read {file}: {e}")
        continue

    if "Rrs" not in ds:
        print(f"No Rrs in {file}")
        continue

    try:
        
        ds_selected = ds["Rrs"].sel(wavelength=wavelengths, method="nearest")
        rrs_full = ds_selected.values  # (lat, lon, bands)

        
        valid_mask = ~np.isnan(rrs_full).any(axis=2)
        rrs_400 = rrs_full[:, :, idx_400]
        rrs_500 = rrs_full[:, :, idx_500]
        value_mask = rrs_500 > rrs_400
        final_mask = valid_mask & value_mask

        
        rrs_valid = rrs_full[final_mask]  # shape: (N, bands)

        if rrs_valid.shape[0] == 0:
            print("No valid pixels after filtering.")
            continue

        
        out_name = os.path.splitext(file)[0] + "_processed.npy"
        out_path = os.path.join(output_folder, out_name)
        np.save(out_path, rrs_valid)
        all_rrs_list.append(rrs_valid)
        print(f"Saved: {out_name}, shape: {rrs_valid.shape}")

    except Exception as e:
        print(f"Error processing {file}: {e}")
        continue


if all_rrs_list:
    merged_rrs = np.vstack(all_rrs_list)
    np.save(os.path.join(output_folder, "all_rrs_array.npy"), merged_rrs)
    print("Saved merged file. Shape:", merged_rrs.shape)
else:
    print("No data saved. All files were empty or invalid.")
