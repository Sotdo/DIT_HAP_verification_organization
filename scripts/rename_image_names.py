# %% ------------------------------------ Import libraries ------------------------------------ #
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from utils import verificationMetadata, roundConfig
from rename_functions import rename_images_per_round
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

# %%
