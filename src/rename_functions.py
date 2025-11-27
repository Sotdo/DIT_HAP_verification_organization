# %% ------------------------------------ Import libraries ------------------------------------ #
import sys
from pathlib import Path
from tqdm import tqdm
from shutil import copyfile
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent))
from utils import verificationMetadata, roundConfig

# %% ------------------------------------ Dataclasses ------------------------------------ #
@dataclass
class renameFormatConfig:
    gene_num: str | int
    gene_name: str
    # sysID: str
    day_or_selection_marker: str
    colony_id: str
    date: str

    def __post_init__(self):
        self.format_string: str = (
            f"{self.gene_num}_"
            f"{self.gene_name}_"
            f"{self.day_or_selection_marker}_"
            f"{self.colony_id}_"
            f"{self.date}"
        )



# %% ------------------------------------ Functions ------------------------------------ #
def rename_image_name(
    subfolder_name: str,
    input_folder_path: Path,
    output_folder_path: Path,
    num2gene: dict,
):
    """Rename image files in the specified folder according to the defined format."""
    image_files = list(input_folder_path.glob("*"))
    if subfolder_name != "replica":
        element_idx = {
            "date": 0,
            "gene_num": 1,
            "colony_id": 2,
            "day_or_selection_marker": subfolder_name
        }
    else:
        element_idx = {
            "date": 0,
            "gene_num": 2,
            "colony_id": 3,
            "day_or_selection_marker": "YHZAY2A"
        }
    for image_file in tqdm(image_files):
        parts = image_file.stem.split("_")
        file_suffix = image_file.suffix
        gene_num = int(parts[element_idx.get("gene_num", 1)])
        gene_name = num2gene.get(gene_num, "UnknownGene")
        day_or_selection_marker = element_idx.get("day_or_selection_marker", "UnknownDayOrMarker")
        colony_id = parts[int(element_idx.get("colony_id", 2))]
        try:
            colony_id = "#" + str(int(colony_id))
        except ValueError:
            colony_id = colony_id
        date = parts[int(element_idx.get("date", 0))]

        rename_config = renameFormatConfig(
            gene_num=gene_num,
            gene_name=gene_name,
            day_or_selection_marker=day_or_selection_marker,
            colony_id=colony_id,
            date=date,
        )

        new_file_name = f"{rename_config.format_string}{file_suffix}"
        new_file_path = output_folder_path / new_file_name
        copyfile(image_file, new_file_path)

def rename_images_per_round(
    round: str,
    verification_metadata: verificationMetadata,
):
    """Rename image files for a specific round based on the provided configuration."""
    round_configuration = roundConfig(
        round_folder_name=round
    )
    for subfolder_name, input_output_paths in round_configuration.all_sub_folders.items():
        input_folder_path = input_output_paths["input"]
        output_folder_path = input_output_paths["output"]

        rename_image_name(
            subfolder_name=subfolder_name,
            input_folder_path=input_folder_path,
            output_folder_path=output_folder_path,
            num2gene=verification_metadata.num2gene,
)
        
