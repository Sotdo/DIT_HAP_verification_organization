# %% ============================= Imports =============================
import sys
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from dataclasses import dataclass

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from tetrad_genotype import genotyping_pipeline

# %% ============================= Parameters =============================
DAY_IMAGE_COLUMNS = {
    3: "3d_image_path",
    4: "4d_image_path",
    5: "5d_image_path",
    6: "6d_image_path"
}

MARKER_IMAGE_COLUMN = "HYG_image_path"
ROUND0_MARKER_IMAGE_COLUMN = "NAT_image_path"

# %% ============================= Data Classes =============================
@dataclass
class configuration:
    data_df: pd.DataFrame
    pdf_output_path: Path
    table_output_path: Path
    log_file: Path = Path("../logs/batch_genotyping.log")

# %% ============================= Constants =============================
SPECIFIC_CASES = SPECIFIC_CASES = [('2nd_round', 57, 'exo2', '#1', 202411),
 ('6th_round', 92, 'zrt1', '#1', 202411),
 ('6th_round', 92, 'zrt1', '#3', 202411),
 ('6th_round', 93, 'fkh2', '#1', 202411),
 ('8th_round', 111, 'trr1', '#2', 202411),
 ('8th_round', 111, 'trr1', '#3', 202411),
 ('13th_round', 165, 'apl2', '#1', 202412),
 ('13th_round', 165, 'apl2', '#2', 202412),
 ('13th_round', 168, 'rpl15', '#3', 202412),
 ('16th_round', 220, 'big1', '#2', 202504),
 ('17th_round', 243, 'hal4', '#1', 202510),
 ('17th_round', 243, 'hal4', '#2', 202510),
 ('18th_round', 306, 'SPAC12G12.16c', '#2', 202511),
 ('18th_round', 309, 'ccr1', '#1', 202511),
 ('18th_round', 309, 'ccr1', '#2', 202511),
 ('19th_round', 327, 'rpl15', '#1', 202511),
 ('19th_round', 327, 'rpl15', '#2', 202511),
 ('19th_round', 328, 'rpl2802', '#1', 202511),
 ('19th_round', 328, 'rpl2802', '#2', 202511),
 ('20th_round', 348, 'big1', '#1', 202511),
 ('20th_round', 348, 'big1', '#2', 202511),
 ('21th_round', 351, 'etr1', '#1', 202512),
 ('23th_round', 262, 'pka1', '#1', 202512),
 ('23th_round', 262, 'pka1', '#2', 202512)]


# %% ============================= Main =============================
logger.info("Starting batch genotyping process...")

logger.info("Loading all images dataframe...")
all_images_df = pd.read_excel("../results/all_combined_all_rounds_crop_summary_manual_annotated_with_genotyping_20260125.xlsx")
# nonE_images_df = all_images_df.query("Kept == 'YES' and (verification_essentiality != 'E' or Comments != '')")
nonE_or_commented_E_images_df = all_images_df.query("Kept == 'YES' and (verification_phenotype != 'E' or Comments.notna())")
# hard_condition_df = all_images_df.query("Kept == 'YES' and (verification_phenotype != 'E' or Comments.notna()) and round != '22th_round' and Genotyping != 'YES'")
# processing_failed_df = nonE_or_commented_E_images_df.query("Genotyping != 'YES' and round != '1st_round' and round != '22th_round'")
# sampled_failed_df = processing_failed_df.sample(n=20, random_state=42)

# %%

# %%
# bad_output_strain_data = pd.merge(
#     all_images_df,
#     bad_output_strain,
#     on=["gene_num", "colony_id"],
#     how="inner"
# )
# %%

config = configuration(
    data_df=nonE_or_commented_E_images_df,
    pdf_output_path=Path("../results/nonE_strain_or_commented_E_genotyping_results.pdf"),
    table_output_path=Path("../results/nonE_strain_or_commented_E_genotyping_results.xlsx"),
    log_file=Path("../logs/batch_genotyping_nonE_strain_or_commented_E.log")
)

# config = configuration(
#     data_df=hard_condition_df,
#     pdf_output_path=Path("../results/hard_condition_genotyping_results.pdf"),
#     table_output_path=Path("../results/hard_condition_genotyping_results.xlsx"),
#     log_file=Path("../logs/batch_genotyping_hard_condition.log")
# )

#  %% ============================= Logging Setup =============================
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{module: <20}</cyan>:<cyan>{line: <4}</cyan> - | "
           "<level>{message}</level>",
    level="INFO"
)

logger.add(
    str(config.log_file),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module: <20}:{line: <4} - | {message}",
    level="DEBUG",
    mode="w"
)
# %%
genotyping_failed = []
all_colony_regions = {}
n = 0
logger.info("Beginning genotyping for each colony...")
with PdfPages(config.pdf_output_path) as pdf:
    for idx, row in tqdm(config.data_df.iterrows(), total=len(config.data_df), desc="Batch Genotyping"):
        round = row["round"]
        gene_num = row["gene_num"]
        gene_name = row["gene_name"]
        gene_essentiality = row["gene_essentiality"]
        phenotype_category = row["phenotype_categories"]
        phenotype_description = row["phenotype_descriptions"]
        colony_id = row["colony_id"]
        date = row["date"]
        image_info = (round, gene_num, gene_name, colony_id, date)

        tetrad_image_paths = {}
        for day, col in DAY_IMAGE_COLUMNS.items():
            img_path = row[col]
            if pd.notna(img_path):
                tetrad_image_paths[day] = Path(img_path)
        if round == "22th_round":
            marker_image_path = row[ROUND0_MARKER_IMAGE_COLUMN]
            if gene_num >= 422:
                expected_columns = 12
            else:
                expected_columns = 13
        else:
            marker_image_path = row[MARKER_IMAGE_COLUMN]
            expected_columns = 12

        if pd.isna(marker_image_path) or not Path(marker_image_path).exists() or image_info in SPECIFIC_CASES:
            logger.warning(f"Marker image not found for round {round}, gene_num {gene_num}, gene {gene_name}, colony {colony_id}. Try genotyping by colony size.")
            try:
                colony_regions, fig = genotyping_pipeline(
                        tetrad_image_paths=tetrad_image_paths,
                        marker_image_path=None,
                        image_info=image_info,
                        colony_columns=expected_columns
                )
                fig.suptitle(
                        f"Round {round} | Gene {gene_num} ({gene_name}, {gene_essentiality}) | "
                        f"Colony {colony_id} | Date {date}\n"
                        f"Phenotype: {phenotype_category} - {phenotype_description}",
                        fontsize=24, y=1.2, fontweight='bold'
                    )

                pdf.savefig(fig, bbox_inches='tight', dpi=150)
                plt.close(fig)
                all_colony_regions[image_info] = colony_regions
            except Exception as e:
                logger.error(f"Genotyping failed for round {round}, gene_num {gene_num}, gene {gene_name}, colony {colony_id}. Error: {e}")
                genotyping_failed.append((*image_info, str(e)))
                continue
        else:
            try:
                marker_image_path = Path(marker_image_path)
                colony_regions, fig = genotyping_pipeline(
                    tetrad_image_paths=tetrad_image_paths,
                    marker_image_path=marker_image_path,
                    image_info=image_info,
                    colony_columns=expected_columns
                )
                fig.suptitle(
                    f"Round {round} | Gene {gene_num} ({gene_name}, {gene_essentiality}) | "
                    f"Colony {colony_id} | Date {date}\n"
                    f"Phenotype: {phenotype_category} - {phenotype_description}",
                    fontsize=24, y=1.2, fontweight='bold'
                )

                pdf.savefig(fig, bbox_inches='tight', dpi=150)
                plt.close(fig)
                all_colony_regions[image_info] = colony_regions
            except Exception as e:
                logger.error(f"Genotyping failed for round {round}, gene_num {gene_num}, gene {gene_name}, colony {colony_id}. Error: {e}")
                genotyping_failed.append((*image_info, str(e)))
                continue

        n += 1
        # if n > 50:
        #     break    
logger.info("Batch genotyping process completed.")
logger.error(f"Genotyping failed for {len(genotyping_failed)} colonies.")
for fail_info in genotyping_failed:
    logger.error(f"- Round {fail_info[0]}, Gene Num {fail_info[1]}, Gene Name {fail_info[2]}, Colony ID {fail_info[3]}, Date {fail_info[4]}. Reason: {fail_info[5]}")
# %%
concated_genotyping_results = pd.concat(all_colony_regions).rename_axis(
    index=["round", "gene_num", "gene_name", "colony_id", "date", "row", "col"]
).reset_index()
concated_genotyping_results.to_excel(config.table_output_path, index=False)

# %%
# tmp = pd.read_excel("../results/batch_genotyping_results.xlsx")
# space_statistics = tmp.groupby(["gene_name", "colony_id", "date"]).apply(
#     lambda df: 
#     pd.Series(
#         {
#             "x_spacing": (df["grid_point_x_day3"].max() - df["grid_point_x_day3"].min()) / 12,
#             "y_spacing": (df["grid_point_y_day3"].max() - df["grid_point_y_day3"].min()) / 4
#         }
#     )
# )
# %%
# all_res = pd.read_excel("../results/all_rounds_combined_verification_summary.xlsx")
# # %%
# not_tetrated = []
# for i in range(48, 351,1):
#     if i not in all_res["gene_num"].values:
#         not_tetrated.append(i)
# %%