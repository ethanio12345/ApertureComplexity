import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from complexity.PyComplexityMetric import (
    PyComplexityMetric,
    MeanAreaMetricEstimator,
    AreaMetricEstimator,
    ApertureIrregularityMetric,
)
from complexity.misc import ModulationIndexScore
from complexity.dicomrt import RTPlan

if __name__ == "__main__":
    # Path to DICOM RTPLAN file - IMRT/VMAT
    # pfile = "RP.dcm"
    from pathlib import Path
    base_path_d4 = Path(r"V:\01 Physics Clinical QA\06 Patient QA\Delta4 Store\DS AXB D4")
    base_path = Path(r"V:\01 Physics Clinical QA\06 Patient QA\ArcCHECK")
    # path_to_rtplan_file = base_path / "RP.1.2.246.352.71.5.76690441797.3290602.20241113123849.dcm"

    d4_rp_files = list(base_path_d4.glob("**/RP.*.dcm"))
    ac_rp_files = list(base_path.glob("**/RT_Plan*.dcm"))

    rp_files = ac_rp_files

    file_name = os.path.join(base_path, f"plan_complexity_summary.csv")
    with open(file_name, "w") as f:
        # write header
        metrics_list = [
            #ModulationIndexScore,
            PyComplexityMetric,
            MeanAreaMetricEstimator,
            AreaMetricEstimator,
            ApertureIrregularityMetric,
        ]
        units = ["CI [mm^-1]", "mm^2", "mm^2", "dimensionless"]
        header = "plan, parent, " + ", ".join([f"{m.__name__} ({u})" for m, u in zip(metrics_list, units)])
        f.write(header + "\n")

        # write results
        for path_to_rtplan_file in tqdm(rp_files):
            # Getting planning data from DICOM file.
            plan_info = RTPlan(filename=path_to_rtplan_file)
            plan_dict = plan_info.get_plan()
            row = [path_to_rtplan_file.stem, path_to_rtplan_file.parent.parent.name]

            # compute per plan
            for cc in metrics_list:
                cc_obj = cc()
                plan_metric = cc_obj.CalculateForPlan(None, plan_dict)
                row.append(str(plan_metric))
            f.write(", ".join(row) + "\n")
