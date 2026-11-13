from wptinference.bootstrap import BootstrapResult, bootstrap_ci, estimate
from wptinference.gaussian import (
    Counterfactual,
    brenier_matrix,
    parallel_transport,
    pt_counterfactual,
)
from wptinference.synthetic import GaussianDesign, random_design, random_spd

__all__ = [
    "BootstrapResult",
    "Counterfactual",
    "GaussianDesign",
    "bootstrap_ci",
    "brenier_matrix",
    "estimate",
    "parallel_transport",
    "pt_counterfactual",
    "random_design",
    "random_spd",
]


def main() -> None:
    """Entry point: see ``wptinference --help``."""
    from wptinference.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
