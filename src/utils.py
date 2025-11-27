# %% ------------------------------------ Import libraries ------------------------------------ #
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd

# %% ------------------------------------ Data classes ------------------------------------ #
@dataclass
class verificationMetadata:

    excel_table_path: Path = Path("../resource/all_for_verification_genes_by_round.xlsx")

    round_column_name: str = "Round"
    gene_num_column_name: str = "Num"
    systematic_id_column_name: str = "SysID"

    gene_IDs_names_products_table_path: Path = Path("../resource/gene_IDs_names_products/20251001_gene_IDs_names_products.tsv")

    deletion_library_essentiality_table_path: Path = Path(
        "../resource/Hayles_2013_OB_merged_categories_sysIDupdated.xlsx"
    )

    def __post_init__(self):
        self.verification_genes: pd.DataFrame = pd.concat(
            pd.read_excel(self.excel_table_path, sheet_name=None), ignore_index=True
        )[["Round", "Num", "SysID"]].sort_values(by=["Num", "Round"])

        gene_IDs_names_products_df = pd.read_csv(
            self.gene_IDs_names_products_table_path, sep="\t"
        )
        gene_IDs_names_products_df["gene_name"] = gene_IDs_names_products_df["gene_name"].fillna(
            gene_IDs_names_products_df["gene_systematic_id"]
        )
        id2name = dict(
            zip(
                gene_IDs_names_products_df["gene_systematic_id"],
                gene_IDs_names_products_df["gene_name"]
            )
        )

        self.verification_genes["Gene"] = self.verification_genes["SysID"].map(id2name)
        deletion_library_essentiality_df = pd.read_excel(
            self.deletion_library_essentiality_table_path
        )

        self.verification_genes = self.verification_genes.merge(
            deletion_library_essentiality_df[[
                "Updated_Systematic_ID",
                "Gene dispensability. This study",
                "One or multi basic phenotypes",
                "Category",
                "Deletion mutant phenotype description"
            ]],
            left_on="SysID",
            right_on="Updated_Systematic_ID",
            how="left"
        ).drop(columns=["Updated_Systematic_ID"]).rename(
            columns={
                "Gene dispensability. This study": "DeletionLibrary_essentiality",
                "One or multi basic phenotypes": "Phenotype_type",
                "Category": "Category",
                "Deletion mutant phenotype description": "Phenotype_description"
            }
        )

        self.num2gene = dict(
            zip(
                self.verification_genes["Num"],
                self.verification_genes["Gene"]
            )
        )

@dataclass
class roundConfig:
    raw_data_folder_path: Path = Path("/hugedata/YushengYang/DIT_HAP_verification/data/raw_data/DIT_HAP_deletion")
    round_folder_name: str = "1st_round"

    days_folder_names: list[str] = field(
        default_factory=lambda: ["3d", "4d", "5d", "6d"]
    )

    replicates_folder_names: str = "replica"

    output_folder_path: Path = Path("/hugedata/YushengYang/DIT_HAP_verification/data/processed_data/DIT_HAP_deletion")

    def __post_init__(self):
        round_folder = self.raw_data_folder_path / self.round_folder_name
        output_folder = self.output_folder_path / self.round_folder_name

        self.all_sub_folders = dict()
        for sub_folder in self.days_folder_names + [self.replicates_folder_names]:
            input_sub_folder = round_folder / sub_folder
            output_sub_folder = output_folder / sub_folder
            output_sub_folder.mkdir(parents=True, exist_ok=True)
            
            self.all_sub_folders[sub_folder] = {
                "input": input_sub_folder,
                "output": output_sub_folder
            }


# %%
