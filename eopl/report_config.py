"""Section selection and plotting options for generated EOPL reports."""
from typing import Literal, TypedDict


class ReportParams(TypedDict, total=False):
    report_format: Literal["py", "ipynb"]
    generate_metadata: bool
    generate_ccdb: bool
    generate_input: bool
    generate_breakpoints: bool
    generate_final: bool
    generate_custom_analysis: bool
    print_all_bkp_data: bool
    plot_1d_images: bool
    plot_1d_indexes: tuple[int, ...]
    plot_2d_images: bool
    plot_2d_indexes: tuple[int, ...]


def default_report_params() -> ReportParams:
    """Return a fresh configuration showing only final outputs and their 2D plots.

    The same selection applies to ordinary, named-run comparison and precision
    analysis reports. Enable extra sections explicitly when they are useful.
    """
    return {
        "report_format": "py",
        "generate_metadata": False,
        "generate_ccdb": False,
        "generate_input": False,
        "generate_breakpoints": False,
        "generate_final": True,
        "generate_custom_analysis": False,
        "print_all_bkp_data": False,
        "plot_1d_images": False,
        "plot_1d_indexes": (0,),
        "plot_2d_images": True,
        "plot_2d_indexes": (0, 0, 0),
    }
