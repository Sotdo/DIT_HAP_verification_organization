# %% ------------------------------------ Imports ------------------------------------ %%
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from skimage import io
# %% ---------------------------------- Plot Settings ---------------------------------- %%
plt.rcParams.update({
    'axes.unicode_minus': True,  # Correctly display minus sign
    'figure.dpi': 300,            # Image resolution
    'axes.titlesize': 20,         # Title font size
    'axes.titleweight': 'bold', # Title font weight
    'axes.labelsize': 18,         # Axis label font size
    'axes.labelweight': 'bold',  # Axis label font weight
    'font.family': 'Arial',       # Set global font to Arial
    'font.size': 16,              # Base font size
    'legend.fontsize': 16,        # Legend font size
    'axes.spines.top': False,      # Remove top spine
    'axes.spines.right': False,     # Remove right spine
    'axes.linewidth': 3,   # Axis line width
    'lines.linewidth': 3,  # Line width
    'lines.solid_joinstyle': 'round',  # Line join style
    'lines.solid_capstyle': 'round',    # Line cap style
    'image.interpolation': 'nearest',  # Image interpolation
    'pdf.compression': 9  # PDF compression level (0-9)
})
# %% ---------------------------------- Configs & Data Classes ---------------------------------- %%
@dataclass
class config:
    verification_result_file: Path = Path("../results/all_combined_all_rounds_crop_summary_manual_annotated_with_genotyping_20260104.xlsx")
    colony_size_result_file: Path = Path("../results/nonE_strain_or_commented_E_genotyping_results.xlsx")

    def __post_init__(self):
        self.verification_result: pd.DataFrame = pd.read_excel(self.verification_result_file).query("Genotyping == 'YES' ").set_index(["gene_num", "gene_name", "round", "colony_id","date"])
        self.colony_size_result: pd.DataFrame = pd.read_excel(self.colony_size_result_file, index_col=[1,2,0,3,4])
# %% ------------------------------------ Functions ------------------------------------ %%
def tetrad_plate_plot(genes, cfg, relative_fraction_per_day, filtered_genotyping_results):
    gene_index = cfg.verification_result[cfg.verification_result.index.get_level_values("gene_name").isin(genes)].index
    # Sort by gene order in the list
    selected_colony = gene_index.to_frame().assign(
        gene_order=lambda df: df['gene_name'].map({gene: i for i, gene in enumerate(genes)})
    ).sort_values('gene_order').drop('gene_order', axis=1).set_index(gene_index.names).index
    fig, axes = plt.subplots(len(selected_colony), 6, figsize=(30, 3 * len(selected_colony)))
    for row, colony in enumerate(selected_colony):
        for col, col_image in enumerate(["3d", "4d", "5d", "6d", "HYG"]):
            image_path = cfg.verification_result.loc[colony, f"{col_image}_image_path"]
            image = io.imread(image_path)
            if col_image != "HYG":
                axes[row, col].imshow(image, cmap="gray")
                area_fraction = relative_fraction_per_day.loc[colony, f"area_day{col+3}"]
                axes[row, col].set_title(f"area fraction (deletion/WT): {area_fraction}")
                day=col + 3
                colony_coordinates = cfg.colony_size_result.loc[colony]
                for idx, region in colony_coordinates.iterrows():
                    cx, cy = region[f"grid_point_x_day{day}"], region[f"grid_point_y_day{day}"]
                    genotype = region["genotype"]
                    color = 'green' if genotype == "WT" else 'red'
                    circle = Circle((cx, cy), 30, edgecolor=color, facecolor='none', linewidth=2, alpha=0.4)
                    axes[row, col].add_patch(circle)
            else:
                axes[row, col].imshow(image)
                axes[row, col].set_title("_".join(map(str, colony)) + "\nHYG Plate")
            axes[row, col].axis("off")
        
        ax_area = axes[row, -1]
        colony_regions = filtered_genotyping_results.loc[colony]
        area_stat = colony_regions.set_index("genotype", append=True).filter(like="area_day")
        area_stat["area_day0"] = 0
        area_stat = area_stat.rename_axis("day", axis=1).stack().reset_index().rename(columns={0: "area"})
        area_stat["day_num"] = area_stat["day"].str.extract(r'day(\d+)').astype(int)
        last_day = area_stat["day_num"].max()
        # area_stat = area_stat.groupby(["col", "day"]).filter(lambda x: x.query("genotype == 'WT'").shape[0]/x.shape[0] == 0.5)
        last_day_WT_colonies_area_mean = area_stat.query("genotype == 'WT' and day_num == @last_day")["area"].median()
        area_stat["area[normalized]"] = area_stat["area"] / last_day_WT_colonies_area_mean
        area_stat = area_stat.query("genotype in ['WT', 'Deletion']")
        sns.lineplot(x="day_num", y="area[normalized]", hue="genotype", data=area_stat, ax=ax_area, palette={"WT": "green", "Deletion": "red"}, errorbar=("pi", 50), estimator="median")
        WT_count = colony_regions.query("genotype == 'WT'")["genotype"].count()
        deletion_count = colony_regions.query("genotype == 'Deletion'")["genotype"].count()
        ax_area.set_title(f"Colony Area Over Time:\n WT ({WT_count}) vs Deletion ({deletion_count})")
        ax_area.axhline(1, color='gray', linestyle='--')
            
    plt.tight_layout()
    plt.show()
    plt.close()

# %% ------------------------------------ Main Function ------------------------------------ %%
cfg = config()
index = cfg.verification_result.index
filtered_genotyping_results = cfg.colony_size_result.loc[index].groupby(["gene_num", "gene_name", "round", "colony_id","date", "col"]).filter(lambda x: x.query("genotype == 'WT'").shape[0]/x.shape[0] == 0.5)
area_table = filtered_genotyping_results.set_index("genotype", append=True).filter(like="area_day")
area_table.fillna(0, inplace=True)
area_table_unstack = area_table.groupby(by=area_table.index.names).median().unstack(level="genotype").reorder_levels([1,0], axis=1)
relative_fraction_per_day = (area_table_unstack["Deletion"] / area_table_unstack["WT"]).round(3)
# %% ------------------------------------ Example Visualization ------------------------------------ %%
selected_colony = pd.MultiIndex.from_tuples([
    (np.int64(61), 'oct1', '3rd_round', '#1', np.int64(202411)),
    (np.int64(144), 'tsr402', '12th_round', '#1', np.int64(202411)),
    (np.int64(116), 'mde4', '8th_round', '#2', np.int64(202411)),
    (np.int64(100), 'rps401', '7th_round', '#1', np.int64(202411)),
    (np.int64(289), 'gmh4', '17th_round', '#2', np.int64(202511)),
    (np.int64(235), 'pmc3', '17th_round', '#1', np.int64(202510)),
    (np.int64(60), 'gcn5', '3rd_round', '#1', np.int64(202411)),
    (np.int64(232), 'csn5', '17th_round', '#3', np.int64(202510)),
    (np.int64(228), 'hul6', '17th_round', '#3', np.int64(202511)),
    (np.int64(227), 'mam2', '17th_round', '#3', np.int64(202511)),
    (np.int64(295), 'SPCC24B10.18', '17th_round', '#3', np.int64(202511))
], names=['gene_num', 'gene_name', 'round', 'colony_id', 'date'])
tetrad_plate_plot(selected_colony, cfg, relative_fraction_per_day, filtered_genotyping_results)
# %% ------------------------------------ small colonies visualization ------------------------------------ %%
small_colonies_genes = [
    "rps801",
    "rpl15",
    "rps1802",
    "rpl2802",
    "cid14",
    "ilv3",
    "plc1",
    "SPAC3F10.16c",
    "gos1",
    "ifa38",
    "anp1",
    "tyr1",
]

tetrad_plate_plot(small_colonies_genes, cfg, relative_fraction_per_day, filtered_genotyping_results)
# %% ------------------------------------ WT colonies visualization ------------------------------------ %%
WT_genes = [
    "gpd1",
    "tna1",
    "urh2",
    "SPAPB24D3.06c",
    "dea2",
    "bmc1",
    "ste11",
    "alr2",
    "gcn2",
    "wtf21",
    "str2",
    "sgp1",
    "coq10",
    "wdr70",
    "oxa102",
    "dar2",
    "big1",
    "ggt1",
    "SPCC1672.01",
    "SPAPB17E12.09",
    "sdg1",
    "usp108",
    "med9",
    "rmi1",
    "taf3",
    "slx8",
    "gcv3",
    "usp103",
    "rsc7",
    "nse3",
]

tetrad_plate_plot(WT_genes, cfg, relative_fraction_per_day, filtered_genotyping_results)
# %% ------------------------------------- WT2nonWT colonies visualization ------------------------------------ %%
WT2nonWT_noised_genes = [
    "pyp3",
    "mam2",
    "hul6",
    "fub1",
    "csn5",
    "rdl1",
    "ymr1",
    "meu31",
    "msn5",
    "SPAC4C5.03",
    "SPAC977.03",
    "mal1",
    "num1",
    "dbl8",
    "fmo1",
    "erm2",
    "thp1",
    "pmc3"
]

tetrad_plate_plot(WT2nonWT_noised_genes, cfg, relative_fraction_per_day, filtered_genotyping_results)
# %%
