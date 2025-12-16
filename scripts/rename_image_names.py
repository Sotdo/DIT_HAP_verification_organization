# %% ------------------------------------ Import libraries ------------------------------------ #
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from utils import verificationMetadata, roundConfig
from rename_functions import rename_images_per_round
from table_organizer import TableConfig, process_all_rounds
# %% ------------------------------------ Main script ------------------------------------ #

verification_meta = verificationMetadata()

default_round_config = roundConfig()

all_rounds = list(default_round_config.raw_data_folder_path.iterdir())
for round_path in all_rounds:
    if round_path.is_dir():
        round_name = round_path.name
        print(f"Processing round: {round_name}")

        rename_images_per_round(
            round=round_name,
            verification_metadata=verification_meta,
        )

table_config = TableConfig(
    image_base_path=Path("/hugedata/YushengYang/DIT_HAP_verification/data/processed_data/DIT_HAP_deletion"),
    table_output_path=Path("../results/all_rounds_renamed_summary.xlsx"),
    image_column_order=["3d", "4d", "5d", "6d", "YHZAY2A"]
)

process_all_rounds(table_config)

# %%
