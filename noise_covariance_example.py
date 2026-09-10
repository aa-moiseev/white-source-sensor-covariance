"""
This example script demonstrates construction of sensor-level noise covariance
induced by uniformly distributed, randomly oriented and uncorrelated brain
sources. It uses precalculated sensor covariance matrix and forward
solutions of real MEG data, which were obtained using standard routines available
in the MNE Python software suite.

The code is tested under MNE Python version 1.9.0.

"""
from pathlib import Path
import sys

import commentjson as cjson
import matplotlib.pyplot as plt
import mne
from packaging.version import Version

from compute_swn_cov import compute_swn_cov

MIN_MNE_VERSION = "1.9.0"

if Version(mne.__version__) < Version(MIN_MNE_VERSION):
    raise RuntimeError(
        f"mne >= {MIN_MNE_VERSION} is required, found {mne.__version__}."
    )

CONFIG_FILE = Path(__file__).with_name("noise_covariance_example.json")

def main():
    with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
        config = cjson.load(config_file)

    (
        dataset_path,       # Path to the M/EEG recording .fif
        org_cov_path,       # Path to the measured sensor covariance .fif
        fwd_path,           # Patth to the forward solutions .fif 
        swn_cov_path,       # Path to the output noise covariance .fif
        org_topo_path,      # Output data covariance topo map .png
        swn_cov_plot_path,  # The noise covariance array plot .png
        swn_eig_path,       # The noise covariance eigenvalues plot .png
        swn_topo_path,      # The noise covariance topo map .png
    ) = covariance_paths(config)

    missing_paths = [
        path for path in (dataset_path, org_cov_path, fwd_path) if not path.is_file()
    ]

    if missing_paths:
        formatted_paths = "\n".join(f"  {path}" for path in missing_paths)
        raise FileNotFoundError(f"Required input file(s) not found:\n{formatted_paths}")

    output_action = "overwrite"

    if swn_cov_path.exists():
        output_action = cov_exists_dlg(swn_cov_path)
        if output_action == "abort":
            print("Calculation aborted; existing output was not changed.")
            return

    verbose = config["verbose"]
    org_cov = mne.read_cov(org_cov_path, verbose=verbose)
    fwd = mne.read_forward_solution(fwd_path, ordered=True, verbose=verbose)
    info = mne.io.read_info(dataset_path, verbose=verbose)

    if config["plot_org_topo"]:
        topomap_config = config["cov_topomap"].copy()
        topomap_config["scalings"] = config["compute_covariance"]["scalings"]
        org_topomap_config = config["adjust_topo_plot"].copy()
        org_topomap_config["clim"] = config["org_cov_topo_clim"]
        show_org_topo = topomap_config["show"]
        topomap_config["show"] = False
        org_topo = org_cov.plot_topomap(info, **topomap_config, verbose=verbose)
        adjust_topo_plot(org_topo, org_topomap_config)
        org_topo.savefig(org_topo_path, dpi=config["dpi"])

        if show_org_topo:
            wait_until_closed()

    if output_action == "skip":
        # Use previously saved calculation result
        swn_cov = mne.read_cov(swn_cov_path, verbose=verbose)
        print(f"Using existing SWN covariance: {swn_cov_path}")
    else:
        # ------------------------------
        # calculate the noise covariance
        # ------------------------------
        swn_cov_data, _, rank, pz = compute_swn_cov(
            fwd, org_cov.data, **config["compute_swn_cov"]
        )
        print(f"SWN covariance: rank = {rank}, pz = {pz}")

        swn_cov = org_cov.copy()
        swn_cov["data"] = swn_cov_data
        swn_cov.save(swn_cov_path, overwrite=True, verbose=verbose)
        print(f"Saved SWN covariance to: {swn_cov_path}")

    if config["plot_swn_cov"]:
        cov_plot_config = config["cov_plot"].copy()
        show_plots = cov_plot_config["show"]
        cov_plot_config["show"] = False

        if config["plot_swn_eig"]:
            covariance_figure, svd_figure = swn_cov.plot(info, **cov_plot_config)
            adjust_cov_plot(covariance_figure, config["adjust_cov_plot"])
            adjust_svd_plot(svd_figure, config["adjust_svd_plot"])
            svd_figure.savefig(swn_eig_path, dpi=config["dpi"])
        else:
            cov_plot_config["show_svd"] = False
            covariance_figure, _ = swn_cov.plot(info, **cov_plot_config)
            adjust_cov_plot(covariance_figure, config["adjust_cov_plot"])

        covariance_figure.savefig(swn_cov_plot_path, dpi=config["dpi"])

        if show_plots:
            wait_until_closed()

            if config["plot_swn_eig"]:
                wait_until_closed()

    if config["plot_swn_topo"]:
        topomap_config = config["cov_topomap"].copy()
        topomap_config["scalings"] = config["compute_covariance"]["scalings"]
        show_swn_topo = topomap_config["show"]
        topomap_config["show"] = False
        swn_topo = swn_cov.plot_topomap(info, **topomap_config, verbose=verbose)
        adjust_topo_plot(swn_topo, config["adjust_topo_plot"])
        swn_topo.savefig(swn_topo_path, dpi=config["dpi"])

        if show_swn_topo:
            wait_until_closed()

    # --- end of the main script ---

def covariance_paths(config):
    """Return the input, output, and plot paths for this example."""
    data_dir = Path(config["data_root"]) / config["subject"]
    dataset_name = Path(config["dsname"]).stem
    return (
        data_dir / config["dsname"],
        data_dir / f"{dataset_name}-org-cov.fif",
        data_dir / f"{dataset_name}-fwd.fif",
        data_dir / f"{dataset_name}-swn-cov.fif",
        data_dir / f"{dataset_name}-org-topo.png",
        data_dir / f"{dataset_name}-swn-cov.png",
        data_dir / f"{dataset_name}-swn-eig.png",
        data_dir / f"{dataset_name}-swn-topo.png",
    )

def adjust_cov_plot(figure, config):
    axis, colorbar = figure.get_axes()

    if config["title"] is not None:
        axis.set_title(config["title"])

    if config["ytitle"] is not None:
        axis.set_ylabel(config["ytitle"])

    image = axis.images[0]
    lower, upper = image.get_clim()

    if config["clim"][0] is not None:
        lower = config["clim"][0]

    if config["clim"][1] is not None:
        upper = config["clim"][1]

    image.set_clim(lower, upper)

    axis.set_title(axis.get_title(), fontsize=config["title_font_size"])
    axis.set_xlabel(axis.get_xlabel(), fontsize=config["axis_title_font_size"])
    axis.set_ylabel(axis.get_ylabel(), fontsize=config["axis_title_font_size"])
    axis.tick_params(axis="both", labelsize=config["ticks_font_size"])

    if config["bar_title"] is not None:
        colorbar.set_title(config["bar_title"])

    colorbar.tick_params(axis="both", labelsize=config["ticks_font_size"])
    figure.set_size_inches(*config["figsize"])


def adjust_svd_plot(figure, config):
    axis = figure.get_axes()[0]
    for line in axis.get_lines():
        if line.get_label() == "_child0":
            line.set_linestyle("")
            line.set_marker(config["marker"])
            line.set_markersize(config["marker_size"])

        if line.get_label() == "_child1":
            line.remove()
            break

    for text in axis.texts:
        if "rank" in text.get_text():
            text.remove()
            break

    lower, upper = axis.get_ylim()

    if config["ylim"][0] is not None:
        lower = config["ylim"][0]

    if config["ylim"][1] is not None:
        upper = config["ylim"][1]

    axis.set_ylim(lower, upper)

    if config["title"] is not None:
        axis.set_title(config["title"])

    if config["ytitle"] is not None:
        axis.set_ylabel(config["ytitle"])

    axis.set_title(axis.get_title(), fontsize=config["title_font_size"])
    axis.set_xlabel(axis.get_xlabel(), fontsize=config["axis_title_font_size"])
    axis.set_ylabel(axis.get_ylabel(), fontsize=config["axis_title_font_size"])
    axis.tick_params(axis="both", labelsize=config["ticks_font_size"])
    figure.set_size_inches(*config["figsize"])


def adjust_topo_plot(figure, config):
    axis, colorbar = figure.get_axes()

    if config["title"] is not None:
        axis.set_title(config["title"], fontsize=config["title_font_size"])

    image = axis.images[0]
    lower, upper = image.get_clim()

    if config["clim"][0] is not None:
        lower = config["clim"][0]

    if config["clim"][1] is not None:
        upper = config["clim"][1]

    image.set_clim(lower, upper)
    colorbar.set_title(colorbar.get_title(), fontsize=config["title_font_size"])
    colorbar.tick_params(axis="both", labelsize=config["ticks_font_size"])
    figure.set_size_inches(*config["figsize"])

    if config["yshift"] is not None:
        for plot_axis in (axis, colorbar):
            bounds = list(plot_axis.get_position().bounds)
            bounds[1] += config["yshift"]
            plot_axis.set_position(bounds)

def wait_until_closed():
    """Block until currentgiven figure's window is closed by the user."""
    plt.show(block=True)


def cov_exists_dlg(output_path):
    """Ask whether an existing output should be reused, replaced, or preserved."""
    prompt = (
        f"SWN covariance already exists: {output_path}\n"
        "Choose [s]kip and use it, [o]verwrite it, or [a]bort: "
    )
    choices = {"s": "skip", "skip": "skip", "o": "overwrite", "overwrite": "overwrite",
               "a": "abort", "abort": "abort"}
    while True:
        try:
            choice = input(prompt).strip().lower()
        except EOFError as error:
            raise RuntimeError("No response received; calculation aborted.") from error

        if choice in choices:
            return choices[choice]

        print("Please enter skip, overwrite, or abort.")


if __name__ == "__main__":
    main()

