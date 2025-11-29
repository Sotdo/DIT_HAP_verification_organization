"""
PDF generation for DIT-HAP verification pipeline.
Creates formatted PDF documents compiling cropped images with metadata.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
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

    # PDF layout parameters
    page_size: str = "A4"  # "letter" or "A4"
    margin: float = 0.5  # inches
    image_width: float = 1.2  # inches for individual images
    image_height: float = 0.9  # inches for individual images
    spacing: float = 0.1  # inches between elements

    # Content parameters
    title_font_size: int = 16
    header_font_size: int = 12
    text_font_size: int = 10
    max_rows_per_page: int = 4

    # Quality thresholds for image inclusion
    min_image_width: int = 100
    min_image_height: int = 100
    max_file_size_mb: float = 10.0

    def __post_init__(self):
        # Ensure output directory exists
        self.output_base_path.mkdir(parents=True, exist_ok=True)

        # Set page size
        if self.page_size.lower() == "letter":
            self.page_width, self.page_height = letter
        else:
            self.page_width, self.page_height = A4


# %% ------------------------------------ Helper Functions ------------------------------------ #
@logger.catch
def get_page_dimensions(config: PDFGeneratorConfig) -> Tuple[float, float]:
    """Get usable page dimensions in inches."""
    usable_width = (config.page_width - 2 * config.margin * inch) / inch
    usable_height = (config.page_height - 2 * config.margin * inch) / inch
    return usable_width, usable_height


@logger.catch
def create_styles(config: PDFGeneratorConfig):
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

    text_style = ParagraphStyle(
        'CustomText',
        parent=styles['Normal'],
        fontSize=config.text_font_size,
        spaceAfter=3
    )

    return {
        'title': title_style,
        'header': header_style,
        'text': text_style
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
            logger.warning(f"Image does not exist: {image_path}")
            return False

        # Check file size
        file_size_mb = image_path.stat().st_size / (1024 * 1024)
        if file_size_mb > config.max_file_size_mb:
            logger.warning(f"Image too large: {image_path} ({file_size_mb:.1f}MB)")
            return False

        # Check image dimensions
        with PILImage.open(image_path) as img:
            width, height = img.size
            if width < config.min_image_width or height < config.min_image_height:
                logger.warning(f"Image too small: {image_path} ({width}x{height})")
                return False

        return True

    except Exception as e:
        logger.error(f"Error validating image {image_path}: {e}")
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
def create_gene_info_row(
    df: pd.DataFrame,
    gene_num: str,
    gene_name: str,
    colony_id: str,
    date: str
) -> List[str]:
    """
    Create gene information row for PDF table.

    Args:
        df: Verification table DataFrame
        gene_num: Gene number
        gene_name: Gene name
        colony_id: Colony identifier
        date: Date string

    Returns:
        List of information items for the first column
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
        return [gene_num, gene_name, "", "", "", colony_id, date]

    record = matching_records.iloc[0]

    return [
        gene_num,
        gene_name,
        record.get('gene_essentiality', ''),
        record.get('phenotype_categories', ''),
        record.get('phenotype_descriptions', '')[:50] + '...' if len(str(record.get('phenotype_descriptions', ''))) > 50 else record.get('phenotype_descriptions', ''),
        colony_id,
        date
    ]


@logger.catch
def create_pdf_page(
    doc: SimpleDocTemplate,
    styles: Dict,
    df: pd.DataFrame,
    gene_records: List[Dict],
    config: PDFGeneratorConfig
):
    """
    Create a single PDF page with gene information and images.

    Args:
        doc: SimpleDocTemplate object
        styles: Dictionary of paragraph styles
        df: Verification table DataFrame
        gene_records: List of gene records for this page
        config: PDF generator configuration
    """
    story = []

    # Create table structure: info + images
    # Columns: gene_info (1), time_points (4), replica_markers (5)
    column_names = ['Gene Info', '3d', '4d', '5d', '6d', 'YES', 'HYG', 'NAT', 'LEU', 'ADE']
    col_widths = [2.5, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]  # inches

    # Create data for table
    table_data = []

    for record in gene_records:
        gene_num = record['gene_num']
        gene_name = record['gene_name']
        colony_id = record['colony_id']
        date = record['date']

        # Gene info column
        gene_info = create_gene_info_row(df, gene_num, gene_name, colony_id, date)
        gene_info_text = "<br/>".join([str(info) if info else "" for info in gene_info])
        row_data = [Paragraph(gene_info_text, styles['text'])]

        # Time point and replica image columns
        for col in column_names[1:]:  # Skip 'Gene Info'
            image_path = get_image_for_gene_colony(df, gene_num, gene_name, colony_id, date, col)

            if image_path and validate_image(image_path, config):
                # Create image with proper sizing
                try:
                    img = Image(str(image_path), width=config.image_width * inch, height=config.image_height * inch)
                    row_data.append(img)
                except Exception as e:
                    logger.error(f"Error adding image {image_path}: {e}")
                    row_data.append(Paragraph("Image Error", styles['text']))
            else:
                row_data.append(Paragraph("No Image", styles['text']))

        table_data.append(row_data)

    # Create table
    if table_data:
        table = Table(table_data, colWidths=[w * inch for w in col_widths])

        # Style the table
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('PADDING', (0, 0), (-1, -1), 6),
        ])

        table.setStyle(style)
        story.append(table)
        story.append(Spacer(1, config.spacing * inch))

    return story


@logger.catch
def generate_verification_pdf(
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

    styles = create_styles(config)

    # Group records by gene and colony
    if df.empty:
        logger.warning(f"No data found for round {round_name}")
        return output_path

    # Create unique gene-colony combinations
    unique_records = df[['gene_num', 'gene_name', 'colony_id', 'date']].drop_duplicates()

    # Group for pages (max_rows_per_page records per page)
    gene_groups = []
    for i in range(0, len(unique_records), config.max_rows_per_page):
        group = unique_records.iloc[i:i + config.max_rows_per_page].to_dict('records')
        gene_groups.append(group)

    logger.info(f"Creating PDF with {len(gene_groups)} pages for round {round_name}")

    # Build document
    story = []

    # Add title
    story.append(Paragraph(f"DIT-HAP Verification Summary - Round {round_name}", styles['title']))
    story.append(Spacer(1, 0.25 * inch))

    # Add summary statistics
    total_genes = len(unique_records)
    complete_records = len(df.dropna(subset=['3d_image_path', '4d_image_path', '5d_image_path', '6d_image_path']))

    summary_text = f"""
    Total Gene-Colony Records: {total_genes}<br/>
    Records with Complete Time Series: {complete_records}<br/>
    Completion Rate: {(complete_records / total_genes * 100):.1f}%
    """
    story.append(Paragraph(summary_text, styles['text']))
    story.append(Spacer(1, 0.25 * inch))

    # Add pages with gene data
    for i, gene_group in enumerate(gene_groups):
        if i > 0:  # Add page break between groups
            story.append(Spacer(1, 0.5 * inch))

        page_story = create_pdf_page(doc, styles, df, gene_group, config)
        story.extend(page_story)

    # Build PDF
    doc.build(story)

    logger.success(f"Generated verification PDF: {output_path}")
    return output_path


# %% ------------------------------------ Main Functions ------------------------------------ #
@logger.catch
def generate_round_pdfs(config: PDFGeneratorConfig, round_name: Optional[str] = None):
    """
    Generate PDFs for specified round or all rounds.

    Args:
        config: PDF generator configuration
        round_name: Specific round to process, or None for all rounds
    """
    # Find available verification tables
    table_base_path = config.processed_data_base_path.parent.parent / "table_structures" / "DIT_HAP_deletion"

    if not table_base_path.exists():
        logger.error(f"Table structures directory not found: {table_base_path}")
        return

    if round_name:
        # Process specific round
        csv_file = table_base_path / f"{round_name}_verification_table.csv"
        if csv_file.exists():
            logger.info(f"Processing round: {round_name}")
            df = pd.read_csv(csv_file)
            generate_verification_pdf(df, config, round_name)
        else:
            logger.error(f"Verification table not found: {csv_file}")
    else:
        # Process all rounds
        csv_files = list(table_base_path.glob("*_verification_table.csv"))
        if not csv_files:
            logger.error("No verification tables found")
            return

        for csv_file in sorted(csv_files):
            round_from_file = csv_file.stem.replace('_verification_table', '')
            try:
                logger.info(f"Processing round: {round_from_file}")
                df = pd.read_csv(csv_file)
                generate_verification_pdf(df, config, round_from_file)
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
                continue

    logger.success("PDF generation completed")


@logger.catch
def create_sample_table_for_testing(config: PDFGeneratorConfig, round_name: str):
    """
    Create a sample verification table for testing PDF generation.

    Args:
        config: PDF generator configuration
        round_name: Name of the experimental round
    """
    # Create sample data
    sample_data = []

    # Add sample records for testing
    sample_genes = [
        ("123", "adh1", "A", "20251128"),
        ("124", "cdc2", "B", "20251128"),
        ("125", "wee1", "A", "20251128")
    ]

    for gene_num, gene_name, colony_id, date in sample_genes:
        record = {
            'gene_num': gene_num,
            'gene_name': gene_name,
            'round': round_name,
            'systematic_id': f'SPAC{gene_num.zfill(4)}',
            'gene_essentiality': 'Essential',
            'phenotype_categories': 'Cell cycle',
            'phenotype_descriptions': 'Deletion affects cell division',
            'colony_id': colony_id,
            'date': date,
            '3d_image_path': f'/mock/path/{gene_num}_{gene_name}_3d_{colony_id}_{date}.png',
            '4d_image_path': f'/mock/path/{gene_num}_{gene_name}_4d_{colony_id}_{date}.png',
            '5d_image_path': f'/mock/path/{gene_num}_{gene_name}_5d_{colony_id}_{date}.png',
            '6d_image_path': f'/mock/path/{gene_num}_{gene_name}_6d_{colony_id}_{date}.png',
            'YES_image_path': f'/mock/path/{gene_num}_{gene_name}_YES_{colony_id}_{date}.png',
            'HYG_image_path': f'/mock/path/{gene_num}_{gene_name}_HYG_{colony_id}_{date}.png',
            'NAT_image_path': f'/mock/path/{gene_num}_{gene_name}_NAT_{colony_id}_{date}.png',
            'LEU_image_path': f'/mock/path/{gene_num}_{gene_name}_LEU_{colony_id}_{date}.png',
            'ADE_image_path': f'/mock/path/{gene_num}_{gene_name}_ADE_{colony_id}_{date}.png'
        }
        sample_data.append(record)

    df = pd.DataFrame(sample_data)

    # Save sample table
    output_path = config.processed_data_base_path.parent.parent / "table_structures" / "DIT_HAP_deletion"
    output_path.mkdir(parents=True, exist_ok=True)

    sample_file = output_path / f"{round_name}_sample_verification_table.csv"
    df.to_csv(sample_file, index=False)

    logger.info(f"Created sample verification table: {sample_file}")
    return sample_file


if __name__ == "__main__":
    # Example usage
    config = PDFGeneratorConfig()

    # Create sample data for testing
    sample_file = create_sample_table_for_testing(config, "1st_round")

    # Generate PDF from sample data
    df = pd.read_csv(sample_file)
    generate_verification_pdf(df, config, "1st_round")

    # Process all rounds (if verification tables exist)
    # generate_round_pdfs(config)