import numpy as np
from prp.lca import run_lca_avg

def optimize_threshold(input_series, relevant_output_indices, correct_response_idx,
                       thresholds=np.arange(0.5, 2.0, 0.05),
                       ITI=0.5, n_repeats=100):
    """
    Finds the threshold z that maximizes reward rate: acc / (ITI + RT)
    """
    best_rr = -np.inf
    best_threshold = None

    for z in thresholds:
        rt, choice = run_lca_avg(
            input_series=input_series,
            relevant_output_indices=relevant_output_indices,
            threshold=z,
            n_repeats=n_repeats
        )

        if rt is None:
            continue

        acc = int(choice == correct_response_idx)
        rr = acc / (ITI + rt)

        if rr > best_rr:
            best_rr = rr
            best_threshold = z

    return best_threshold
