#!/usr/bin/env python3
"""pdf2md - Extract tables, figures, and vector schematics from a PDF and
convert it to LLM-friendly Markdown.

This is the CLI version of the original extraction script. It keeps the
padded-table / caption-driven figure extraction logic, but exposes every
parameter as a command-line flag and writes outputs next to the input by
default.

Usage examples:
    # simplest form - outputs <stem>.md and <stem>_images/ next to the PDF
    pdf2md paper.pdf

    # custom output locations and rendering quality
    pdf2md paper.pdf -o out/paper.md --img-dir out/figs --dpi 300

    # batch mode - process several PDFs in one invocation
    pdf2md a.pdf b.pdf c.pdf --img-subdir imgs

    # disable pymupdf multiprocessing (useful inside containers)
    pdf2md paper.pdf --no-multiprocessing

Run `pdf2md --help` for the full flag reference.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

# Heavy imports (pymupdf, pymupdf4llm) are deferred until a PDF is actually
# processed. This keeps `--help` / `--version` fast and dependency-free, and
# gives a clean error message instead of an ImportError trace if the runtime
# deps aren't installed yet.
_pymupdf = None  # type: ignore
_pymupdf4llm = None  # type: ignore


def _load_backends() -> None:
    """Lazily import pymupdf and pymupdf4llm once, on first use."""
    global _pymupdf, _pymupdf4llm
    if _pymupdf is not None:
        return
    try:
        import pymupdf  # noqa: F401  (re-exported below)
        import pymupdf4llm  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "pdf2md requires `pymupdf` and `pymupdf4llm`.\n"
            "Install them with:  pip install pymupdf pymupdf4llm\n"
            f"(root cause: {exc})"
        ) from exc
    _pymupdf = pymupdf
    _pymupdf4llm = pymupdf4llm


__version__ = "0.1.0"

log = logging.getLogger("pdf2md")


# ---------------------------------------------------------------------------
# Core extraction logic (preserved from the original script)
# ---------------------------------------------------------------------------


def extract_padded_elements(
    doc,
    output_img_dir: Path,
    vertical_margin: float = 25.0,
    dpi: int = 150,
    min_fig_size: tuple = (50, 50),
) -> None:
    """Detect tables, raster images, and vector diagrams (schematics/circuitry),
    expanding bounding boxes to capture surrounding captions and headings.
    """
    _load_backends()
    pymupdf = _pymupdf
    output_img_dir.mkdir(parents=True, exist_ok=True)

    for page_num, page in enumerate(doc):
        text_blocks = page.get_text("blocks")
        page_rect = page.rect

        # =====================================================================
        # 1. TABLE EXTRACTION
        # =====================================================================
        table_rects: list[pymupdf.Rect] = []
        try:
            tables = page.find_tables()
            for idx, table in enumerate(tables):
                table_rect = pymupdf.Rect(table.bbox)
                table_rects.append(table_rect)
                padded_rect = pymupdf.Rect(table_rect)

                padded_rect.y0 = max(0, table_rect.y0 - vertical_margin)
                padded_rect.y1 = min(page_rect.height, table_rect.y1 + vertical_margin)

                for block in text_blocks:
                    bx0, by0, bx1, by1, text, _, _ = block
                    block_rect = pymupdf.Rect(bx0, by0, bx1, by1)
                    text_clean = text.strip()

                    if 0 <= (table_rect.y0 - by1) <= 40:
                        padded_rect.include_rect(block_rect)

                    if (
                        0 <= (by0 - table_rect.y1) <= 40
                        or 0 <= (table_rect.y0 - by1) <= 40
                    ) and text_clean.lower().startswith("table"):
                        padded_rect.include_rect(block_rect)

                padded_rect.x0 = max(0, padded_rect.x0 - 10)
                padded_rect.x1 = min(page_rect.width, padded_rect.x1 + 10)

                pix = page.get_pixmap(clip=padded_rect, dpi=dpi)
                pix.save(str(output_img_dir / f"table_p{page_num + 1}_{idx + 1}.png"))
        except Exception as exc:
            # Fallback if complex table layout rendering crashes the parser thread
            log.debug("Table extraction failed on page %d: %s", page_num + 1, exc)

        # =====================================================================
        # 2. FIGURE & VECTOR SCHEMATIC EXTRACTION
        # =====================================================================
        candidate_fig_rects: list[pymupdf.Rect] = []

        # A) Embedded raster images
        for img_info in page.get_image_info():
            r = pymupdf.Rect(img_info["bbox"])
            if r.width >= min_fig_size[0] and r.height >= min_fig_size[1]:
                candidate_fig_rects.append(r)

        # B) Vector graphic clusters (schematics, pinouts, block diagrams)
        try:
            vector_clusters = page.cluster_drawings()
            for r in vector_clusters:
                # Filter out tiny line artifacts, full-page borders, and tables
                if (
                    min_fig_size[0] <= r.width <= page_rect.width * 0.95
                    and min_fig_size[1] <= r.height <= page_rect.height * 0.85
                ):
                    # Skip anything overlapping an already-extracted table
                    if not any(t_rect.intersects(r) for t_rect in table_rects):
                        candidate_fig_rects.append(r)
        except Exception as exc:
            log.debug("Vector clustering failed on page %d: %s", page_num + 1, exc)

        # C) Caption-driven matching (matches "Figure X" captions with diagram boxes)
        fig_count = 1
        processed_captions: set[int] = set()

        for block in text_blocks:
            bx0, by0, bx1, by1, text, _, block_no = block
            text_clean = text.strip()

            if text_clean.lower().startswith(("figure ", "fig.", "diagram ", "scheme ")):
                caption_rect = pymupdf.Rect(bx0, by0, bx1, by1)
                processed_captions.add(block_no)

                # Search region above and below caption for graphics
                search_region = pymupdf.Rect(
                    0,
                    max(0, by0 - 300),
                    page_rect.width,
                    min(page_rect.height, by1 + 300),
                )

                matching_rects = [
                    r for r in candidate_fig_rects if search_region.intersects(r)
                ]

                if matching_rects:
                    # Combine diagram bounds + caption bounds into one rectangle
                    final_rect = pymupdf.Rect(caption_rect)
                    for r in matching_rects:
                        final_rect.include_rect(r)
                else:
                    # Fallback: if no vector cluster formed, capture box above caption
                    final_rect = pymupdf.Rect(
                        max(0, bx0 - 15),
                        max(0, by0 - 220),
                        min(page_rect.width, bx1 + 15),
                        by1 + 10,
                    )

                # Expand slightly for padding
                final_rect.x0 = max(0, final_rect.x0 - 10)
                final_rect.x1 = min(page_rect.width, final_rect.x1 + 10)
                final_rect.y0 = max(0, final_rect.y0 - 10)
                final_rect.y1 = min(page_rect.height, final_rect.y1 + 10)

                pix = page.get_pixmap(clip=final_rect, dpi=dpi)
                pix.save(
                    str(output_img_dir / f"figure_p{page_num + 1}_{fig_count}.png")
                )
                fig_count += 1

        # D) Extract standalone images without explicit "Figure" caption text
        for r in candidate_fig_rects:
            already_captured = any(t_rect.intersects(r) for t_rect in table_rects)
            if not already_captured and fig_count == 1:
                padded = pymupdf.Rect(
                    max(0, r.x0 - 10),
                    max(0, r.y0 - vertical_margin),
                    min(page_rect.width, r.x1 + 10),
                    min(page_rect.height, r.y1 + vertical_margin),
                )
                pix = page.get_pixmap(clip=padded, dpi=dpi)
                pix.save(
                    str(output_img_dir / f"figure_p{page_num + 1}_{fig_count}.png")
                )
                fig_count += 1


# ---------------------------------------------------------------------------
# Per-PDF processing
# ---------------------------------------------------------------------------


def process_pdf(
    input_pdf: Path,
    output_md: Path | None,
    img_dir: Path | None,
    *,
    vertical_margin: float = 25.0,
    dpi: int = 150,
    min_fig_width: float = 50.0,
    min_fig_height: float = 50.0,
    workers: int | None = 4,
) -> tuple[Path, Path]:
    """Process a single PDF: extract padded figures/tables, then render Markdown.

    Returns (output_md_path, img_dir_path) for reporting.
    """
    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    # Default output paths live next to the input PDF
    if output_md is None:
        output_md = input_pdf.with_suffix(".md")
    if img_dir is None:
        img_dir = input_pdf.parent / f"{input_pdf.stem}_images"

    output_md.parent.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    _load_backends()
    pymupdf = _pymupdf
    pymupdf4llm = _pymupdf4llm

    # Optional PyMuPDF multiprocessing control (guarded for cross-version compat)
    if workers is not None and hasattr(pymupdf, "set_multiprocessing"):
        try:
            pymupdf.set_multiprocessing(workers=workers)
        except Exception as exc:  # pragma: no cover - environment specific
            log.debug("set_multiprocessing failed: %s", exc)

    log.info("Opening %s", input_pdf)
    doc = pymupdf.open(str(input_pdf))

    try:
        log.info(
            "Extracting padded tables, figures, and vector schematics -> %s", img_dir
        )
        extract_padded_elements(
            doc=doc,
            output_img_dir=img_dir,
            vertical_margin=vertical_margin,
            dpi=dpi,
            min_fig_size=(min_fig_width, min_fig_height),
        )

        log.info("Generating Markdown structure -> %s", output_md)
        markdown_content = pymupdf4llm.to_markdown(
            doc=doc,
            write_images=False,
            dpi=dpi,
            page_markers=True,
            page_separators=True,
        )
        output_md.write_text(markdown_content, encoding="utf-8")
    finally:
        doc.close()

    log.info("Done: %s", input_pdf.name)
    return output_md, img_dir


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf2md",
        description=(
            "Extract padded tables, figures, and vector schematics from a PDF and "
            "convert it to LLM-friendly Markdown (via pymupdf4llm)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        metavar="PDF",
        help="One or more input PDF files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="MD",
        help=(
            "Output Markdown file. Defaults to <input_stem>.md next to each input. "
            "In batch mode this applies to the first file only; for the rest the "
            "default is used."
        ),
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory for extracted images. Defaults to <input_stem>_images/ next "
            "to each input."
        ),
    )
    parser.add_argument(
        "--img-subdir",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "If given, image dirs are created as <parent>/<NAME>/ for each input "
            "(overrides --img-dir). Handy in batch mode to keep all figs together."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Rendering DPI for both image extraction and Markdown generation.",
    )
    parser.add_argument(
        "--vertical-margin",
        type=float,
        default=25.0,
        metavar="PTS",
        help="Vertical padding (in PDF points) added around tables and figures.",
    )
    parser.add_argument(
        "--min-fig-width",
        type=float,
        default=50.0,
        metavar="PTS",
        help="Minimum width (in PDF points) for an image/vector cluster to be kept.",
    )
    parser.add_argument(
        "--min-fig-height",
        type=float,
        default=50.0,
        metavar="PTS",
        help="Minimum height (in PDF points) for an image/vector cluster to be kept.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="PyMuPDF multiprocessing worker count. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-multiprocessing",
        action="store_true",
        help="Disable pymupdf multiprocessing entirely (overrides --workers).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (-v = INFO, -vv = DEBUG).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all logging except errors.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _configure_logging(verbose: int, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    else:
        level = logging.WARNING - (verbose * 10)
        level = max(level, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_img_dir(
    input_pdf: Path, *, img_dir: Path | None, img_subdir: str | None
) -> Path | None:
    if img_subdir is not None:
        return input_pdf.parent / img_subdir
    return img_dir


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose, args.quiet)

    workers: int | None = None if args.no_multiprocessing else args.workers

    failures: list[tuple[Path, BaseException]] = []

    for i, pdf in enumerate(args.inputs):
        # --output only meaningfully applies to the first input in batch mode
        out_md = args.output if i == 0 else None
        img_dir = _resolve_img_dir(pdf, img_dir=args.img_dir, img_subdir=args.img_subdir)

        try:
            md_path, _ = process_pdf(
                input_pdf=pdf,
                output_md=out_md,
                img_dir=img_dir,
                vertical_margin=args.vertical_margin,
                dpi=args.dpi,
                min_fig_width=args.min_fig_width,
                min_fig_height=args.min_fig_height,
                workers=workers,
            )
            print(f"OK   {pdf}  ->  {md_path}")
        except FileNotFoundError as exc:
            log.error("%s", exc)
            failures.append((pdf, exc))
        except Exception as exc:
            log.exception("Failed to process %s", pdf)
            failures.append((pdf, exc))

    if failures:
        log.error("Failed on %d/%d input(s):", len(failures), len(args.inputs))
        for pdf, exc in failures:
            log.error("  - %s: %s", pdf, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())