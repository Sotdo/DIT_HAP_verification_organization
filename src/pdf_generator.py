"""
PDF generation for DIT-HAP verification pipeline.
Creates formatted PDF documents compiling cropped images with metadata.
Refactored to use PILLOW only for simpler logic.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from PIL import Image as PILImage, ImageDraw, ImageFont
from loguru import logger
import re

from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))


# %% ------------------------------------ Configuration ------------------------------------ #
@dataclass
class PDFGeneratorConfig:
    """Simple configuration for PDF generation."""
    # Input/output paths
    processed_data_base_path: Path = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion")
    output_base_path: Path = Path("/hugedata/YushengYang/DIT_HAP_verification/data/merged_pdfs/DIT_HAP_deletion")
    table_structures_path: Path = Path("../results")

    # Page layout (in inches - easy to understand)
    page_width: float = 11.69  # A4 landscape width
    page_height: float = 8.27   # A4 landscape height
    margin: float = 1.0        # Margin around page

    # Content sizing (in inches)
    image_width: float = 1.0   # Width of each image
    image_height: float = 0.4  # Height of each image
    gene_info_width: float = 3.0  # Width of gene info column
    spacing: float = 0.1       # Spacing between elements
    padding: float = 0.05      # Padding inside cells

    # Text settings (in points - standard font sizing)
    title_font_size: int = 16  # Title text size
    header_font_size: int = 12  # Header text size
    text_font_size: int = 9    # Regular text size

    # Quality settings
    dpi: int = 300            # Resolution for print quality

    # Visual style
    line_color: str = "#333333"  # Grid line color (dark gray)
    border_color: str = "#000000" # Outer border color (black)
    background_color: str = "#fafafa"  # Alternating row background

    # Quality thresholds for image inclusion
    min_image_width: int = 100
    min_image_height: int = 100
    max_file_size_mb: float = 10.0

    def __post_init__(self):
        """Initialize fonts and ensure output directory exists."""
        self.output_base_path.mkdir(parents=True, exist_ok=True)

        # Initialize fonts with automatic fallback
        self.title_font = self._load_font(self.title_font_size, bold=True)
        self.header_font = self._load_font(self.header_font_size, bold=True)
        self.text_font = self._load_font(self.text_font_size, bold=False)

    def _load_font(self, size: int, bold: bool = False):
        """Load font with automatic fallback."""
        try:
            # Try Arial first
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            try:
                # Try system fonts
                import os
                if os.name == 'nt':  # Windows
                    font_path = "C:/Windows/Fonts/arial.ttf"
                else:  # Linux/Unix
                    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                return ImageFont.truetype(font_path, size)
            except Exception:
                # Default font as last resort
                logger.warning(f"Using default font for size {size}")
                return ImageFont.load_default()

    # Computed properties (users don't need to set these)
    @property
    def page_width_px(self) -> int:
        """Page width in pixels."""
        return int(self.page_width * self.dpi)

    @property
    def page_height_px(self) -> int:
        """Page height in pixels."""
        return int(self.page_height * self.dpi)

    @property
    def margin_px(self) -> int:
        """Margin in pixels."""
        return int(self.margin * self.dpi)

    @property
    def image_width_px(self) -> int:
        """Image width in pixels."""
        return int(self.image_width * self.dpi)

    @property
    def image_height_px(self) -> int:
        """Image height in pixels."""
        return int(self.image_height * self.dpi)

    @property
    def spacing_px(self) -> int:
        """Spacing in pixels."""
        return int(self.spacing * self.dpi)

    @property
    def padding_px(self) -> int:
        """Padding in pixels."""
        return int(self.padding * self.dpi)


# %% ------------------------------------ Helper Functions ------------------------------------ #
@logger.catch
def extract_colony_number(colony_id: str) -> int:
    """Extract numeric value from colony ID for sorting.

    Args:
        colony_id: Colony identifier (e.g., "A", "B", "1", "2")

    Returns:
        Integer value for sorting
    """
    try:
        if colony_id.isdigit():
            return int(colony_id)
        elif colony_id.isalpha():
            # Convert A=1, B=2, C=3, etc.
            return ord(colony_id.upper()) - ord('A') + 1
        else:
            # Extract numbers from mixed IDs like "colonyA1"
            numbers = re.findall(r'\d+', colony_id)
            return int(numbers[0]) if numbers else 0
    except Exception:
        return 0


@logger.catch
def sort_dataframe_by_gene_and_colony(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame by gene number and colony ID.

    Args:
        df: Input DataFrame

    Returns:
        Sorted DataFrame
    """
    # Convert gene_num to int for proper sorting
    df['gene_num_int'] = df['gene_num'].astype(int)
    df['colony_num'] = df['colony_id'].apply(extract_colony_number)

    # Sort by gene_num first, then by colony_num
    df_sorted = df.sort_values(['gene_num_int', 'colony_num'])

    # Drop sorting columns
    return df_sorted.drop(columns=['gene_num_int', 'colony_num'])


@logger.catch
def validate_image(image_path: Path, config: PDFGeneratorConfig) -> bool:
    """
    Validate that image meets quality requirements for PDF inclusion.

    Args:
        image_path: Path to image file
        config: PDF generator configuration

    Returns:
        True if image is valid for inclusion
    """
    try:
        if not image_path.exists():
            return False

        # Check file size
        file_size_mb = image_path.stat().st_size / (1024 * 1024)
        if file_size_mb > config.max_file_size_mb:
            return False

        # Check image dimensions
        with PILImage.open(image_path) as img:
            width, height = img.size
            if width < config.min_image_width or height < config.min_image_height:
                return False

        return True

    except Exception:
        return False


@logger.catch
def get_image_for_gene_colony(
    df: pd.DataFrame,
    gene_num: str,
    gene_name: str,
    colony_id: str,
    date: str,
    column: str
) -> Optional[Path]:
    """
    Get image path for specific gene, colony, and time point/marker.

    Args:
        df: Verification table DataFrame
        gene_num: Gene number
        gene_name: Gene name
        colony_id: Colony identifier
        date: Date string
        column: Column name (e.g., "3d", "4d", "YES", "HYG")

    Returns:
        Path to image if found, None otherwise
    """
    # Filter for the specific record
    mask = (
        (df['gene_num'] == gene_num) &
        (df['gene_name'] == gene_name) &
        (df['colony_id'] == colony_id) &
        (df['date'] == date)
    )

    matching_records = df[mask]
    if matching_records.empty:
        return None

    record = matching_records.iloc[0]
    image_path = record.get(f"{column}_image_path")

    if pd.isna(image_path) or not image_path:
        return None

    return Path(image_path)


@logger.catch
def calculate_dynamic_page_height(
    gene_records: list[dict],
    config: PDFGeneratorConfig
) -> int:
    """
    Calculate dynamic page height based on number of colonies for a gene.

    Args:
        gene_records: List of gene records for this gene (all have same gene_num)
        config: PDF generator configuration

    Returns:
        Height in pixels
    """
    num_colonies = len(gene_records)

    # Height components (in inches)
    title_height = 1.0      # Space for title and summary
    header_height = 0.4       # Table header
    row_height = max(config.image_height, 0.3) + 0.1  # Image or text height + padding
    content_height = row_height * num_colonies
    spacing = 0.5            # Extra spacing

    # Total height in inches
    total_height_inches = title_height + header_height + content_height + spacing

    # Minimum height of 4 inches
    total_height_inches = max(total_height_inches, 4.0)

    # Convert to pixels
    return int(total_height_inches * config.dpi)


@logger.catch
def draw_text_with_wrapping(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    max_width: int,
    font,
    line_spacing: int = 15
) -> int:
    """
    Draw text with automatic line wrapping.

    Args:
        draw: ImageDraw object
        text: Text to draw
        x: X coordinate
        y: Y coordinate
        max_width: Maximum width before wrapping
        font: Font to use
        line_spacing: Space between lines

    Returns:
        Y coordinate after drawing all text
    """
    # Simple text wrapping by character
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]

        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    # Draw each line
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, fill="black", font=font)
        current_y += line_spacing

    return current_y


@logger.catch
def create_gene_page_pil(
    df: pd.DataFrame,
    gene_records: list[dict],
    config: PDFGeneratorConfig,
    round_name: str,
    page_index: int,
    total_genes: int
) -> Path:
    """
    Create individual PDF page for a specific gene using PILLOW with high quality.

    Args:
        df: Verification table DataFrame
        gene_records: List of gene records for this gene (all have same gene_num)
        config: PDF generator configuration
        round_name: Name of the experimental round
        page_index: Index of this gene page (for multi-page PDF)
        total_genes: Total number of genes

    Returns:
        Path to generated PDF file for this gene
    """
    # Calculate dynamic page height based on colony count
    page_height = calculate_dynamic_page_height(gene_records, config)

    # Create temporary image for this gene
    gene_num = gene_records[0]['gene_num']
    gene_name = gene_records[0]['gene_name']

    temp_output_path = config.output_base_path / f"temp_gene_{gene_num}_page_{page_index}.png"

    # Create high quality image with white background
    img = PILImage.new('RGB', (config.page_width_px, page_height), 'white')
    draw = ImageDraw.Draw(img)

    current_y = config.margin_px

    # Draw title on single line with better formatting
    num_colonies = len(gene_records)

    # Title components
    round_text = f"{round_name}"
    gene_text = f"Gene {gene_num} - {gene_name}"
    page_text = f"Total Colonies: {num_colonies} | Page {page_index + 1} of {total_genes}"

    # Fonts
    regular_font = config.header_font
    bold_font = config.title_font

    # Calculate text widths
    round_width = draw.textlength(round_text, font=regular_font)
    gene_width = draw.textlength(gene_text, font=bold_font)
    page_width = draw.textlength(page_text, font=regular_font)
    gap_width = int(0.4 * config.dpi)  # Gap between sections

    # Calculate starting position for centering
    total_title_width = round_width + gap_width + gene_width + gap_width + page_width
    start_x = (config.page_width_px - total_title_width) // 2

    # Draw title components on single line
    current_x = start_x
    draw.text((current_x, current_y), round_text, font=regular_font, fill="black")
    current_x += round_width + gap_width
    draw.text((current_x, current_y), gene_text, font=bold_font, fill="black")
    current_x += gene_width + gap_width
    draw.text((current_x, current_y), page_text, font=regular_font, fill="black")

    current_y += config.spacing_px

    # Draw horizontal line for separation
    line_y = current_y + int(0.05 * config.dpi)
    draw.line(
        [(config.margin_px, line_y), (config.page_width_px - config.margin_px, line_y)],
        fill=config.border_color, width=3
    )
    current_y = line_y + config.spacing_px

    # Define column structure: gene_info + 4 time_points + 5 replica_markers = 10 columns
    time_point_columns = ['3d', '4d', '5d', '6d']
    replica_columns = ['YES', 'HYG', 'NAT', 'LEU', 'ADE']
    all_columns = ['Gene Info'] + time_point_columns + replica_columns

    # Calculate column widths with safety checks
    usable_width = config.page_width - 2 * config.margin
    gene_info_width_px = int(config.gene_info_width * config.dpi)

    # Ensure usable width is positive and sufficient
    if usable_width <= gene_info_width_px:
        logger.warning(f"Usable width ({usable_width:.1f}) too small for gene info width ({gene_info_width_px}px)")
        # Fall back to reasonable proportions
        gene_info_width_px = int(usable_width * 0.35)  # Use 35% for gene info
        remaining_width = usable_width - gene_info_width_px
    else:
        remaining_width = usable_width - gene_info_width_px

    # Calculate image column width with minimum protection
    image_col_width = max(remaining_width // 9, 50)  # At least 50 pixels per column

    logger.debug(f"Layout: usable={usable_width:.1f}, gene_info={gene_info_width_px}px, image_col={image_col_width}px")

    # Calculate row height
    gene_info_height = 0.3  # inches for gene info text
    row_height = max(config.image_height, gene_info_height) + config.spacing

    # Draw table headers with background
    header_height = 0.3  # inches
    header_height_px = int(header_height * config.dpi)
    current_x = config.margin_px

    # Draw header background
    draw.rectangle(
        [config.margin_px, current_y, config.page_width_px - config.margin_px, current_y + header_height_px],
        fill="#f0f0f0"
    )

    # Draw header text centered in each column
    for col_name in all_columns:
        col_width = gene_info_width_px if col_name == 'Gene Info' else image_col_width
        text_bbox = draw.textbbox((0, 0), col_name, font=config.header_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Center text in column
        text_x = current_x + (col_width - text_width) // 2
        text_y = current_y + (header_height_px - text_height) // 2

        draw.text((text_x, text_y), col_name, fill="black", font=config.header_font)
        current_x += col_width

    current_y += header_height_px + 3  # 3 pixels for grid line

    # Draw table grid and content
    for row_idx, record in enumerate(gene_records):
        record_gene_num = record['gene_num']
        record_gene_name = record['gene_name']
        colony_id = record['colony_id']
        date = record['date']

        # Find matching record in DataFrame for additional info
        mask = (
            (df['gene_num'] == record_gene_num) &
            (df['gene_name'] == record_gene_name) &
            (df['colony_id'] == colony_id) &
            (df['date'] == date)
        )

        matching_records = df[mask]
        if not matching_records.empty:
            record_data = matching_records.iloc[0]
            phenotype_category = record_data.get('phenotype_categories', 'N/A')
            essentiality = record_data.get('gene_essentiality', 'N/A')
        else:
            phenotype_category = 'N/A'
            essentiality = 'N/A'

        # Handle nan/float values and convert to strings
        if pd.isna(phenotype_category) or phenotype_category != phenotype_category:  # nan check
            phenotype_category = 'N/A'
        if pd.isna(essentiality) or essentiality != essentiality:  # nan check
            essentiality = 'N/A'

        # Ensure values are strings
        phenotype_category = str(phenotype_category)
        essentiality = str(essentiality)

        # Row height in pixels
        row_height_px = int(row_height * config.dpi)

        # Draw row background (alternating colors for readability)
        if row_idx % 2 == 0:
            draw.rectangle(
                [config.margin_px, current_y, config.page_width_px - config.margin_px, current_y + row_height_px],
                fill=config.background_color
            )

        # Draw gene info with improved layout
        current_x = config.margin_px + config.padding_px
        current_y_inner = current_y + config.padding_px

        # First line: Gene, Colony, Date
        line1_text = f"Gene: {record_gene_num} - {record_gene_name}   Colony: {colony_id}   Date: {date}"
        draw.text((current_x, current_y_inner), line1_text, font=config.text_font, fill="black")
        current_y_inner += int((10 / 72) * config.dpi)  # 10 points line spacing

        # Second line: Category and Essentiality with bold content
        category_label = "Category:"
        essentiality_label = "Essentiality:"

        # Draw labels and content with bold effect
        label_width = draw.textlength(category_label, font=config.text_font)
        gap_text = "   "
        gap_width = draw.textlength(gap_text, font=config.text_font)

        # Draw "Category:" in regular, content in simulated bold
        draw.text((current_x, current_y_inner), category_label, font=config.text_font, fill="black")
        content_x = current_x + label_width + 5
        draw.text((content_x, current_y_inner), phenotype_category, font=config.text_font, fill="black")
        draw.text((content_x + 1, current_y_inner), phenotype_category, font=config.text_font, fill="black")  # Bold effect

        # Draw essentiality
        essentiality_x = content_x + draw.textlength(f" {phenotype_category}{gap_text}", font=config.text_font) + gap_width
        draw.text((essentiality_x, current_y_inner), essentiality_label, font=config.text_font, fill="black")
        content_x = essentiality_x + draw.textlength(f"{essentiality_label} ", font=config.text_font)
        draw.text((content_x, current_y_inner), essentiality, font=config.text_font, fill="black")
        draw.text((content_x + 1, current_y_inner), essentiality, font=config.text_font, fill="black")  # Bold effect

        current_y_inner += int((10 / 72) * config.dpi)  # 10 points line spacing

        # Draw vertical grid lines
        current_x = config.margin_px + gene_info_width_px
        draw.line(
            [(current_x, current_y), (current_x, current_y + row_height_px)],
            fill=config.line_color, width=2
        )

        # Draw images for each column with centering
        current_x = config.margin_px + gene_info_width_px
        for column in time_point_columns + replica_columns:
            # Draw vertical grid line before column
            draw.line(
                [(current_x, current_y), (current_x, current_y + row_height_px)],
                fill=config.line_color, width=2
            )

            image_path = get_image_for_gene_colony(df, record_gene_num, record_gene_name, colony_id, date, column)

            if image_path and validate_image(image_path, config):
                try:
                    # Load and resize image with high quality, maintaining aspect ratio
                    with PILImage.open(image_path) as img_colony:
                        # Calculate max dimensions that fit in the cell
                        max_width = image_col_width - 2*config.padding_px
                        max_height = row_height_px - 2*config.padding_px

                        # Get original dimensions
                        original_width, original_height = img_colony.size

                        logger.debug(f"Image {image_path.name}: original={original_width}x{original_height}, max_cell={max_width}x{max_height}")

                        # Calculate scaling factor to fit within max dimensions while maintaining aspect ratio
                        if original_width <= 0 or original_height <= 0:
                            logger.error(f"Invalid original dimensions for {image_path}: {original_width}x{original_height}")
                            continue

                        scale_x = max_width / original_width if original_width > 0 else 0
                        scale_y = max_height / original_height if original_height > 0 else 0
                        scale = min(scale_x, scale_y)  # Use the smaller scale to ensure both dimensions fit

                        if scale <= 0:
                            logger.warning(f"Invalid scale {scale} for {image_path}, using minimum size")
                            scale = 0.1  # Minimum scale to prevent zero dimensions

                        logger.debug(f"Scale: {scale:.3f}, calculated size: {int(original_width * scale)}x{int(original_height * scale)}")

                        # Calculate new dimensions with minimum size constraints
                        new_width = int(original_width * scale)
                        new_height = int(original_height * scale)

                        # Ensure minimum sizes to prevent 0-dimension errors
                        min_width = 10  # Minimum 10 pixels
                        min_height = 10  # Minimum 10 pixels
                        new_width = max(new_width, min_width)
                        new_height = max(new_height, min_height)

                        # Ensure we don't exceed cell boundaries
                        new_width = min(new_width, max_width)
                        new_height = min(new_height, max_height)

                        # Final safety check
                        if new_width <= 0 or new_height <= 0:
                            logger.warning(f"Calculated invalid size {new_width}x{new_height} for {image_path}, using minimum")
                            new_width = min_width
                            new_height = min_height

                        logger.debug(f"Final size for {image_path.name}: {new_width}x{new_height}")

                        # Ensure all dimensions are integers for Pillow
                        new_width_int = max(1, int(new_width))  # Minimum 1 pixel
                        new_height_int = max(1, int(new_height))  # Minimum 1 pixel

                        # Resize with high quality maintaining aspect ratio
                        img_colony = img_colony.resize((new_width_int, new_height_int), PILImage.Resampling.LANCZOS)

                        # Center image in cell with integer coordinates
                        img_x = current_x + max(0, (image_col_width - new_width_int) // 2)
                        img_y = current_y + max(0, (row_height_px - new_height_int) // 2)

                        # Ensure paste coordinates are integers and within bounds
                        img_x_int = max(0, int(img_x))
                        img_y_int = max(0, int(img_y))

                        img.paste(img_colony, (img_x_int, img_y_int))
                except Exception as e:
                    logger.error(f"Error adding image {image_path}: {e}")
                    # Draw "Error" text centered with integer coordinates
                    error_text = "Error"
                    text_bbox = draw.textbbox((0, 0), error_text, font=config.text_font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    text_x = current_x + max(0, (image_col_width - text_width) // 2)
                    text_y = current_y + max(0, (row_height_px - text_height) // 2)
                    # Ensure text coordinates are integers
                    text_x_int = int(text_x)
                    text_y_int = int(text_y)
                    draw.text((text_x_int, text_y_int), error_text, fill="red", font=config.text_font)
            else:
                # Draw "No Image" text centered with integer coordinates
                no_image_text = "No Image"
                text_bbox = draw.textbbox((0, 0), no_image_text, font=config.text_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = current_x + max(0, (image_col_width - text_width) // 2)
                text_y = current_y + max(0, (row_height_px - text_height) // 2)
                # Ensure text coordinates are integers
                text_x_int = int(text_x)
                text_y_int = int(text_y)
                draw.text((text_x_int, text_y_int), no_image_text, fill="#808080", font=config.text_font)

            current_x += image_col_width

        # Draw final vertical line
        draw.line(
            [(current_x, current_y), (current_x, current_y + row_height_px)],
            fill=config.line_color, width=2
        )

        # Draw horizontal line for row
        draw.line(
            [(config.margin_px, current_y + row_height_px), (config.page_width_px - config.margin_px, current_y + row_height_px)],
            fill=config.line_color, width=2
        )

        current_y += row_height_px

    # Draw outer border
    draw.rectangle(
        [config.margin_px, config.margin_px, config.page_width_px - config.margin_px, current_y],
        outline=config.border_color, width=3
    )

    # Save as high-quality PNG first (300 DPI)
    img.save(temp_output_path, 'PNG', dpi=(config.dpi, config.dpi), quality=95)
    return temp_output_path


@logger.catch
def create_verification_pdf(
    df: pd.DataFrame,
    config: PDFGeneratorConfig,
    round_name: str
) -> Path:
    """
    Generate verification PDF for a specific round using PILLOW.

    Args:
        df: Verification table DataFrame
        config: PDF generator configuration
        round_name: Name of the experimental round

    Returns:
        Path to generated PDF file
    """
    output_path = config.output_base_path / f"round_{round_name}_verification_summary.pdf"

    # Group records by gene and colony
    if df.empty:
        logger.warning(f"No data found for round {round_name}")
        return output_path

    # Create unique gene-colony combinations and sort by gene_num then colony_id
    unique_records = df[['gene_num', 'gene_name', 'colony_id', 'date']].drop_duplicates()
    unique_records = sort_dataframe_by_gene_and_colony(unique_records)

    # Group by gene_num - one page per gene
    gene_groups = []
    current_gene_num = None
    current_gene_records = []

    for _, record in unique_records.iterrows():
        gene_num = record['gene_num']

        if current_gene_num is None:
            # First record
            current_gene_num = gene_num
            current_gene_records = [record.to_dict()]
        elif gene_num == current_gene_num:
            # Same gene, add to current group
            current_gene_records.append(record.to_dict())
        else:
            # Different gene, save current group and start new one
            if current_gene_records:
                gene_groups.append(current_gene_records)
            current_gene_num = gene_num
            current_gene_records = [record.to_dict()]

    # Add the last gene group
    if current_gene_records:
        gene_groups.append(current_gene_records)

    logger.info(f"Creating PDF with {len(gene_groups)} pages for round {round_name}")

    # Create individual PNG pages for each gene
    temp_png_files = []
    total_genes = len(gene_groups)

    for i, gene_group in tqdm(enumerate(gene_groups), total=total_genes, desc="Generating PDF pages"):
        # logger.info(f"Creating page {i + 1}/{total_genes} for gene {gene_group[0]['gene_num']} ({len(gene_group)} colonies)")

        try:
            temp_png_path = create_gene_page_pil(
                df, gene_group, config, round_name, i, total_genes
            )
            temp_png_files.append(temp_png_path)
        except Exception as e:
            logger.error(f"Error creating PDF page for gene {gene_group[0]['gene_num']}: {e}")
            continue

    # Convert PNGs to PDF if we have files
    if temp_png_files:
        try:
            # Convert all PNGs to RGB if needed and ensure high quality
            rgb_images = []
            for png_path in temp_png_files:
                img = PILImage.open(png_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                rgb_images.append(img)

            # Save as high-quality multipage PDF with proper DPI
            rgb_images[0].save(
                output_path,
                "PDF",
                resolution=config.dpi,  # Use configured DPI (300)
                save_all=True,
                append_images=rgb_images[1:],
                quality=95  # Maximum quality
            )

            # Clean up temporary files
            for temp_png in temp_png_files:
                temp_png.unlink(missing_ok=True)

            logger.success(f"Generated high-quality verification PDF using PILLOW: {output_path}")

        except Exception as e:
            logger.error(f"Error creating PDF with PILLOW: {e}")
            logger.info(f"Individual PNG files remain available in: {config.output_base_path}")
    else:
        logger.warning("No PDF pages created")

    return output_path


# %% ------------------------------------ Main Functions ------------------------------------ #
@logger.catch
def generate_round_pdfs(config: PDFGeneratorConfig, round_name: Optional[str] = None):
    """Generate PDFs for specified round or all rounds."""
    # Find verification table file
    verification_file = config.table_structures_path / "all_rounds_verification_summary.xlsx"

    if not verification_file.exists():
        logger.error(f"Verification table file not found: {verification_file}")
        return

    try:
        # Read all sheets from Excel file
        all_sheets = pd.read_excel(verification_file, sheet_name=None)

        if not all_sheets:
            logger.error("No sheets found in verification table file")
            return

        logger.info(f"Found {len(all_sheets)} sheets in verification table")

        if round_name:
            # Process specific round
            if round_name in all_sheets:
                logger.info("")
                logger.info("-"*30 + f"Processing round: {round_name}" + "-"*30)
                df = all_sheets[round_name]
                create_verification_pdf(df, config, round_name)
            else:
                logger.error(f"Round not found in verification table: {round_name}")
        else:
            # Process all rounds (skip Summary sheet)
            for sheet_name in all_sheets:
                if sheet_name.lower() == 'summary':
                    continue

                try:
                    logger.info("")
                    logger.info("-"*30 + f"Processing round: {sheet_name}" + "-"*30)
                    df = all_sheets[sheet_name]
                    create_verification_pdf(df, config, sheet_name)
                except Exception as e:
                    logger.error(f"Error processing sheet {sheet_name}: {e}")
                    continue

        logger.success("PDF generation completed")

    except Exception as e:
        logger.error(f"Error reading verification table file: {e}")


if __name__ == "__main__":
    # Example usage
    config = PDFGeneratorConfig()

    # Process all rounds (if verification tables exist)
    # generate_round_pdfs(config)