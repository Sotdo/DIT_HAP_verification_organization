# %% ------------------------------------ Import libraries ------------------------------------ #
import sys
from pathlib import Path
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent / "src"))
from utils import verificationMetadata, roundConfig
from rename_functions import rename_images_per_round
from table_organizer import TableConfig, process_all_rounds

# %% ------------------------------------ Logger setup ------------------------------------ #
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
    "../logs/rename_image_names.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module: <20}:{line: <4} - | {message}",
    level="DEBUG",
    mode="w"
)

# %% ------------------------------------ Main script ------------------------------------ #

verification_meta = verificationMetadata()

default_round_config = roundConfig()

all_rounds = list(default_round_config.raw_data_folder_path.iterdir())
not_processed_name = [
    "22th_round",
]
for round_path in all_rounds:
    if round_path.is_dir():
        round_name = round_path.name
        print(f"Processing round: {round_name}")
        if round_name in not_processed_name:
            # print(f"Skipping unrecognized round: {round_name}")
            continue
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
