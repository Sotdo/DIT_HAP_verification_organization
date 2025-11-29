"""
PDF generation for DIT-HAP verification pipeline.
Creates formatted PDF documents compiling cropped images with metadata.
Refactored for cleaner structure and better format handling.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))


# %% ------------------------------------ Configuration ------------------------------------ #
@dataclass
class PDFGeneratorConfig:
    """Configuration for PDF generation."""
    # Input/output paths
    processed_data_base_path: Path = Path("/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion")
    output_base_path: Path = Path("/hugedata/YushengYang/DIT_HAP_verification/data/merged_pdfs/DIT_HAP_deletion")
    table_structures_path: Path = Path("../results")  # Path to table structures output from table_organizer.py

    # PDF layout parameters
    page_size: str = "A4_landscape"  # "letter", "A4", "letter_landscape", "A4_landscape"
    margin: float = 0.25  # inches (smaller margin for landscape)
    image_width: float = 0.9  # inches for individual images (slightly smaller for more columns)
    image_height: float = 0.35  # inches for individual images (maintain aspect ratio)
    spacing: float = 0.05  # inches between elements

    # Content parameters
    title_font_size: int = 16
    header_font_size: int = 12
    text_font_size: int = 9
    max_rows_per_page: int = 4  # 4 rows per page

    # Quality thresholds for image inclusion
    min_image_width: int = 100
    min_image_height: int = 100
    max_file_size_mb: float = 10.0

    def __post_init__(self):
        # Ensure output directory exists
        self.output_base_path.mkdir(parents=True, exist_ok=True)

        # Set page size with landscape support
        page_size_lower = self.page_size.lower()
        if page_size_lower == "letter":
            self.page_width, self.page_height = letter
        elif page_size_lower == "letter_landscape":
            self.page_width, self.page_height = letter[1], letter[0]  # Swap for landscape
        elif page_size_lower == "a4_landscape":
            self.page_width, self.page_height = A4[1], A4[0]  # Swap for landscape
        else:
            self.page_width, self.page_height = A4


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
            import re
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
def create_pdf_styles(config: PDFGeneratorConfig):
    """Create PDF styles for document."""
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=config.title_font_size,
        spaceAfter=12,
        alignment=1  # Center
    )

    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=config.header_font_size,
        spaceAfter=6
    )

    # Gene info style with smaller font
    gene_info_style = ParagraphStyle(
        'GeneInfo',
        parent=styles['Normal'],
        fontSize=config.text_font_size,
        spaceAfter=2,
        leading=10  # Tighter line spacing
    )

    return {
        'title': title_style,
        'header': header_style,
        'gene_info': gene_info_style
    }


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
        from PIL import Image as PILImage

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
def create_gene_info_text(
    df: pd.DataFrame,
    gene_num: str,
    gene_name: str,
    colony_id: str,
    date: str
) -> str:
    """
    Create gene information text for PDF table.

    Args:
        df: Verification table DataFrame
        gene_num: Gene number
        gene_name: Gene name
        colony_id: Colony identifier
        date: Date string

    Returns:
        Formatted text string for gene info
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
        return f"Gene: {gene_num} - {gene_name}<br/>Colony: {colony_id}<br/>Date: {date}<br/>Category: N/A"

    record = matching_records.iloc[0]

    # Build gene info text line by line
    info_lines = [
        f"Gene: {gene_num} - {gene_name}",
        f"Colony: {colony_id}",
        f"Date: {date}",
        f"Category: {record.get('phenotype_categories', 'N/A')}"
    ]

    return "<br/>".join(info_lines)


@logger.catch
def create_verification_pdf(
    df: pd.DataFrame,
    config: PDFGeneratorConfig,
    round_name: str
) -> Path:
    """
    Generate verification PDF for a specific round.

    Args:
        df: Verification table DataFrame
        config: PDF generator configuration
        round_name: Name of the experimental round

    Returns:
        Path to generated PDF file
    """
    output_path = config.output_base_path / f"round_{round_name}_verification_summary.pdf"

    # Create document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=(config.page_width, config.page_height),
        leftMargin=config.margin * inch,
        rightMargin=config.margin * inch,
        topMargin=config.margin * inch,
        bottomMargin=config.margin * inch
    )

    styles = create_pdf_styles(config)

    # Group records by gene and colony
    if df.empty:
        logger.warning(f"No data found for round {round_name}")
        return output_path

    # Create unique gene-colony combinations and sort by gene_num then colony_id
    unique_records = df[['gene_num', 'gene_name', 'colony_id', 'date']].drop_duplicates()

    # Sort by gene_num first, then by colony_id for proper grouping
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

    # Build document
    story = []

    # Add title
    story.append(Paragraph(f"DIT-HAP Verification Summary - Round {round_name}", styles['title']))
    story.append(Spacer(1, 0.25 * inch))

    # Add summary statistics
    total_genes = len(unique_records)
    complete_time_series = len(df.dropna(subset=['3d_image_path', '4d_image_path', '5d_image_path', '6d_image_path'], how='all'))

    summary_text = f"""
    Total Gene-Colony Records: {total_genes}<br/>
    Records with Complete Time Series: {complete_time_series}<br/>
    Completion Rate: {(complete_time_series / total_genes * 100):.1f}%
    """
    story.append(Paragraph(summary_text, styles['gene_info']))
    story.append(Spacer(1, 0.25 * inch))

    # Add pages with gene data
    for i, gene_group in enumerate(gene_groups):
        if i > 0:  # Add page break between groups
            story.append(Spacer(1, 0.5 * inch))

        story.extend(create_pdf_table(doc, styles, df, gene_group, config))

    # Build PDF
    doc.build(story)

    logger.success(f"Generated verification PDF: {output_path}")
    return output_path


@logger.catch
def create_pdf_table(
    doc: SimpleDocTemplate,
    styles: dict,
    df: pd.DataFrame,
    gene_records: list[dict],
    config: PDFGeneratorConfig
):
    """
    Create a PDF table with gene information and images.

    Args:
        doc: SimpleDocTemplate object
        styles: Dictionary of paragraph styles
        df: Verification table DataFrame
        gene_records: List of gene records for this page
        config: PDF generator configuration

    Returns:
        List of flowable elements for the table
    """
    # Define column structure: gene_info + 4 time_points + 5 replica_markers = 10 columns
    time_point_columns = ['3d', '4d', '5d', '6d']
    replica_columns = ['YES', 'HYG', 'NAT', 'LEU', 'ADE']
    all_columns = ['Gene Info'] + time_point_columns + replica_columns

    # Calculate column widths for landscape format - optimized for 10 columns
    usable_width = (config.page_width - 2 * config.margin * inch) / inch

    # Better width distribution for landscape: gene info gets 2.5", images get remaining width/9
    gene_info_width = 2.5
    image_width = (usable_width - gene_info_width) / 9  # 9 image columns (4 time points + 5 replicas)

    # Ensure minimum image width for visibility
    min_image_width = 0.8
    image_width = max(image_width, min_image_width)

    # Total columns: 1 gene info + 9 image columns = 10 columns
    col_widths = [gene_info_width] + [image_width] * 9

    # Create table for this gene (all records have same gene_num)
    current_gene = None
    current_group = []

    for record in gene_records:
        gene_num = record['gene_num']

        if current_gene is None:
            current_gene = gene_num
            current_group = [record]
        elif gene_num == current_gene:
            # Same gene, add to current group
            current_group.append(record)
        else:
            # Different gene, save current group and start new one
            if current_group:
                gene_groups.append(current_group)
            current_gene = gene_num
            current_group = [record]

    # Add the last group
    if current_group:
        gene_groups.append(current_group)

    # Create separate tables for each gene group with spacing
    all_elements = []

    for group_idx, gene_group in enumerate(gene_groups):
        # Create table data for this gene group
        table_data = []
        headers = all_columns
        table_data.append(headers)

        for record in gene_group:
            gene_num = record['gene_num']
            gene_name = record['gene_name']
            colony_id = record['colony_id']
            date = record['date']

            # Create gene info text
            gene_info_text = create_gene_info_text(df, gene_num, gene_name, colony_id, date)
            row_data = [Paragraph(gene_info_text, styles['gene_info'])]

            # Add images for each column
            for column in time_point_columns + replica_columns:
                image_path = get_image_for_gene_colony(df, gene_num, gene_name, colony_id, date, column)

                if image_path and validate_image(image_path, config):
                    try:
                        img = Image(str(image_path), width=config.image_width * inch, height=config.image_height * inch)
                        row_data.append(img)
                    except Exception as e:
                        logger.error(f"Error adding image {image_path}: {e}")
                        row_data.append(Paragraph("Error", styles['gene_info']))
                else:
                    row_data.append(Paragraph("No Image", styles['gene_info']))

            table_data.append(row_data)

        # Create table for this gene group
        if table_data:
            table = Table(table_data, colWidths=[w * inch for w in col_widths])

            # Style table with gene group separation
            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), config.text_font_size),
                ('PADDING', (0, 0), (-1, -1), 3),
                # Add thicker border below the last row of each gene group
                ('LINEBELOW', (0, len(headers) - 1), (-1, len(table_data) - 1), 2, colors.black),
            ])

            table.setStyle(style)
            all_elements.append(table)

            # Add spacing between gene groups (except after the last group)
            if group_idx < len(gene_groups) - 1:
                all_elements.append(Spacer(1, 0.2 * inch))  # 0.2" gap between gene groups

    return all_elements


# %% ------------------------------------ Main Functions ------------------------------------ #
@logger.catch
def generate_round_pdfs(config: PDFGeneratorConfig, round_name: Optional[str] = None):
    """
    Generate PDFs for specified round or all rounds.

    Args:
        config: PDF generator configuration
        round_name: Specific round to process, or None for all rounds
    """
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
                logger.info(f"Processing round: {round_name}")
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
                    logger.info(f"Processing round: {sheet_name}")
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