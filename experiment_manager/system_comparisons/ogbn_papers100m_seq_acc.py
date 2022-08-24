from marius.tools.preprocess.datasets.ogbn_papers100m import OGBNPapers100M
import executor as e
import reporting
from pathlib import Path

BASE_PATH = Path("experiment_manager/system_comparisons/configs/ogbn_papers100m_seq_acc")


def run_ogbn_papers100m_seq_acc(dataset_dir, results_dir, overwrite, enable_dstat, enable_nvidia_smi, show_output, short, num_runs=1):
    """
    Models: GraphSage
    Systems: Marius
    """

    dataset_name = "ogbn_papers100m_seq_acc"

    marius_gs = BASE_PATH / Path("marius_gs.yaml")

    if not (dataset_dir / Path(dataset_name) / Path("edges/train_edges.bin")).exists():
        print("==== Preprocessing {} =====".format(dataset_name))
        dataset = OGBNPapers100M(dataset_dir / Path(dataset_name))
        dataset.download()
        dataset.preprocess(num_partitions=8192, sequential_train_nodes=True)
    else:
        print("==== {} already preprocessed =====".format(dataset_name))

    for i in range(num_runs):
        # Run Marius
        e.run_config(marius_gs, results_dir / Path("ogbn_papers100m_seq_acc/marius_gs"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "marius")

    reporting.print_results_summary([results_dir / Path("ogbn_papers100m_seq_acc/marius_gs")])
