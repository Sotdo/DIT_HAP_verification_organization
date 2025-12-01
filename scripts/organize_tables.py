"""
Main execution script for Task 3: Table structure organization and PDF generation.
Processes cropped images to create organized verification tables and generate summary PDFs.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from table_organizer import (
    TableConfig,
    process_all_rounds as process_table_rounds,
)
from pdf_generator import (
    PDFGeneratorConfig,
    generate_round_pdfs as generate_pdf_rounds,
    generate_pdf_for_given_genes,
)


# %% ------------------------------------ Configuration ------------------------------------ #
@dataclass
class OrganizeTablesConfig:
    """Configuration for table organization and PDF generation."""
    # Base paths
    processed_data_base_path: Path = Path(
        "/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion"
    )
    output_base_path: Path = Path(
        "../results/"
    )

    # Table organization configuration
    table_config: TableConfig = field(
        default_factory=lambda: TableConfig(
            base_path=Path(
                "/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion"
            ),
            output_path=Path(
                "../results"
            )
        )
    )

    # PDF generation configuration
    pdf_config: PDFGeneratorConfig = field(
        default_factory=lambda: PDFGeneratorConfig(
            processed_data_base_path=Path(
                "/hugedata/YushengYang/DIT_HAP_verification/data/cropped_images/DIT_HAP_deletion"
            ),
            output_base_path=Path(
                "../results/merged_pdfs"
            )
        )
    )

    # Processing options
    process_tables: bool = False
    generate_pdfs: bool = True
    create_samples: bool = False
    # rounds_to_process: list[str] = field(default_factory=list)  # Empty = all rounds
    rounds_to_process: list[str] = field(default_factory=lambda: ["18th_round"])  # Empty = all rounds
    gene_nums: list[int] = field(default_factory=lambda: [
            243,
            249,
            255,
            256,
            262,
            265,
            273,
            279,
            280,
            281,
            292,
            296,
            297,
            300,
            307
        ]
    )

    def __post_init__(self):
        # Ensure output directories exist
        self.table_config.output_path.mkdir(parents=True, exist_ok=True)
        self.pdf_config.output_base_path.mkdir(parents=True, exist_ok=True)

        # Set base paths in sub-configurations
        self.table_config.base_path = self.processed_data_base_path
        self.pdf_config.processed_data_base_path = self.processed_data_base_path
        self.pdf_config.table_structures_path = self.table_config.output_path


# %% ------------------------------------ Main Functions ------------------------------------ #
@logger.catch
def process_tables(config: OrganizeTablesConfig):
    """Process verification tables for specified rounds."""
    logger.info("Starting table structure organization...")

    if config.rounds_to_process:
        logger.info(f"Processing specified rounds: {config.rounds_to_process}")
        for round_name in config.rounds_to_process:
            try:
                from table_organizer import process_round_tables
                process_round_tables(round_name, config.table_config)
            except Exception as e:
                logger.error(f"Error processing tables for round {round_name}: {e}")
    else:
        logger.info("Processing all available rounds...")
        process_table_rounds(config.table_config)


@logger.catch
def process_pdfs(config: OrganizeTablesConfig):
    """Generate verification PDFs for specified rounds."""
    logger.info("Starting PDF generation...")

    if config.rounds_to_process:
        logger.info(f"Generating PDFs for specified rounds: {config.rounds_to_process}")
        for round_name in config.rounds_to_process:
            try:
                generate_pdf_rounds(config.pdf_config, round_name)
            except Exception as e:
                logger.error(f"Error generating PDF for round {round_name}: {e}")
    else:
        logger.info("Generating PDFs for all rounds with table structures...")
        generate_pdf_rounds(config.pdf_config)
    
    if config.gene_nums:
        logger.info(f"Generating PDFs for specified gene numbers: {config.gene_nums}")
        try:
            generate_pdf_for_given_genes(config.pdf_config, config.gene_nums)
        except Exception as e:
            logger.error(f"Error generating PDF for gene number {config.gene_nums}: {e}")


@logger.catch
def create_sample_data(config: OrganizeTablesConfig):
    """Create sample data for testing table organization and PDF generation."""
    logger.info("Creating sample data for testing...")

    # Create sample table data
    from pdf_generator import create_sample_table_for_testing
    sample_table_file = create_sample_table_for_testing(config.pdf_config, "1st_round")
    logger.info(f"Created sample table: {sample_table_file}")

    # Create sample PDF data (using the sample table that was just created)
    # PDF will be generated from the sample table data
    logger.info("Sample PDF data ready for generation from sample table")


@logger.catch
def verify_prerequisites(config: OrganizeTablesConfig) -> bool:
    """
    Verify that prerequisites are met for processing.

    Args:
        config: Organize tables configuration

    Returns:
        True if all prerequisites are met
    """
    logger.info("Verifying prerequisites...")

    # Check if processed data directory exists
    if not config.processed_data_base_path.exists():
        logger.error(f"Processed data directory does not exist: {config.processed_data_base_path}")
        return False

    # Check for cropped image subdirectories
    round_dirs = [d for d in config.processed_data_base_path.iterdir() if d.is_dir()]
    if not round_dirs:
        logger.error(f"No round directories found in: {config.processed_data_base_path}")
        return False

    logger.info(f"Found {len(round_dirs)} round directories")

    # Check for required resource files
    resource_files = [
        Path("../resource") / "all_for_verification_genes_by_round.xlsx",
        Path("../resource") / "gene_IDs_names_products" / "20251001_gene_IDs_names_products.tsv",
        Path("../resource") / "Hayles_2013_OB_merged_categories_sysIDupdated.xlsx"
    ]

    for resource_file in resource_files:
        if not resource_file.exists():
            logger.error(f"Required resource file not found: {resource_file}")
            return False

    logger.info("All resource files found")

    # Check for table structures if generating PDFs
    if config.generate_pdfs:
        table_structures_dir = config.table_config.output_path
        if not table_structures_dir.exists():
            if config.process_tables:
                logger.warning("Table structures directory not found, will be created during table processing")
            else:
                logger.error(f"Table structures directory not found: {table_structures_dir}")
                return False
        else:
            csv_files = list(table_structures_dir.glob("*_verification_table.csv"))
            logger.info(f"Found {len(csv_files)} verification table files")

    logger.success("All prerequisites verified")
    return True


@logger.catch
def setup_logging(config: OrganizeTablesConfig):
    """Set up logging for the organization process."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{module:<20}</cyan>:{line:<4} | <level>{message}</level>",
        level="DEBUG"
    )

    # Add file logging
    log_file = config.output_base_path / "table_organization.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        colorize=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {module:<20} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days"
    )

    logger.info("Table organization logging initialized")


@logger.catch
def main():
    """Main function for table organization and PDF generation."""
    # Create configuration
    config = OrganizeTablesConfig()

    # Set up logging
    setup_logging(config)

    logger.info("=" * 80)
    logger.info("Starting DIT-HAP Table Organization and PDF Generation")
    logger.info("=" * 80)

    # Verify prerequisites
    if not verify_prerequisites(config):
        logger.error("Prerequisites not met. Exiting.")
        return

    try:
        # Create sample data if requested
        if config.create_samples:
            create_sample_data(config)
            return

        # Process verification tables
        if config.process_tables:
            process_tables(config)

        # Generate PDFs (after tables are processed)
        if config.generate_pdfs:
            process_pdfs(config)

        logger.success("=" * 80)
        logger.success("Table organization and PDF generation completed successfully!")
        logger.success("=" * 80)

    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")
        raise


@logger.catch
def print_usage():
    """Print usage instructions."""
    print("DIT-HAP Table Organization and PDF Generation Script")
    print("")
    print("This script processes cropped images to create:")
    print("1. Organized verification tables with gene metadata")
    print("2. PDF compilations of verification data")
    print("")
    print("Usage:")
    print("    python scripts/organize_tables.py")
    print("")
    print("Configuration:")
    print("    Modify OrganizeTablesConfig class in this script")
    print("    - process_tables: Enable/disable table processing")
    print("    - generate_pdfs: Enable/disable PDF generation")
    print("    - create_samples: Create sample data for testing")
    print("    - rounds_to_process: List of specific rounds (empty = all)")
    print("")
    print("Requirements:")
    print("    - Cropped images from Task 2")
    print("    - Resource files in resource/ directory")
    print("    - Sufficient disk space for output")


if __name__ == "__main__":
    # Print usage if requested
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        print_usage()
        sys.exit(0)

    # Run main processing
    main()