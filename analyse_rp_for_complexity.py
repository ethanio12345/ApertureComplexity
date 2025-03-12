import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from complexity.PyComplexityMetric import (
    PyComplexityMetric,
    MeanAreaMetricEstimator,
    AreaMetricEstimator,
    ApertureIrregularityMetric,
)
from complexity.misc import ModulationComplexityScore
from complexity.dicomrt import RTPlan

if __name__ == "__main__":
    # Path to DICOM RTPLAN file - IMRT/VMAT
    # pfile = "RP.dcm"
    from pathlib import Path
    base_path_d4 = Path(r"V:\01 Physics Clinical QA\06 Patient QA\Delta4 Store\DS AXB D4")
    base_path_APP = Path(r"P:\8. Staff\EB\DataMiner Re-Build\1000024210")
    base_path_ac = Path("AC Measurements")
    # path_to_rtplan_file = base_path / "RP.1.2.246.352.71.5.76690441797.3290602.20241113123849.dcm"

    d4_rp_files = list(base_path_d4.glob("**/RP.*.dcm"))
    app_rp_files = list(base_path_APP.glob("**/RP.*.dcm"))
    ac_rp_files = list(base_path_ac.glob("**/RT_Plan*.dcm"))

    base_path = base_path_ac
    rp_files = ac_rp_files

    file_name = os.path.join(base_path, f"plan_complexity_summary.csv")

    import logging

    logging.basicConfig(filename='errors.log', level=logging.ERROR)

    with open(file_name, "w") as f:
        # write header
        metrics_list = [
            PyComplexityMetric,
            MeanAreaMetricEstimator,
            AreaMetricEstimator,
            ApertureIrregularityMetric,
            ModulationComplexityScore
        ]
        units = ["CI [mm^-1]", "mm^2", "mm^2", "dimensionless", "unknown units"]
        header = "plan, parent, AUID, Course ID, SOP Instance UID, " + ", ".join([f"{m.__name__} ({u})" for m, u in zip(metrics_list, units)])
        f.write(header + "\n")

        # write results
        for path_to_rtplan_file in tqdm(rp_files):
            try:
                # Getting planning data from DICOM file.
                plan_info = RTPlan(filename=path_to_rtplan_file)
                plan_dict = plan_info.get_plan()
                study_info = plan_info.ds.StudyID
                sop_instance_uid = plan_info.ds.SOPInstanceUID

                row = [
                    path_to_rtplan_file.stem,
                    path_to_rtplan_file.parent.parent.name,
                    str(int(path_to_rtplan_file.parent.parent.name[:10])) if path_to_rtplan_file.parent.parent.name[:10].isdigit() else
                    str(int(path_to_rtplan_file.parent.parent.name[-10:])) if path_to_rtplan_file.parent.parent.name[-10:].isdigit() else "N/A",
                    study_info,
                    sop_instance_uid
                ]

                # compute per plan
                for cc in metrics_list:
                    try:
                        cc_obj = cc()
                        plan_metric = cc_obj.CalculateForPlan(None, plan_dict)
                        row.append(str(plan_metric))
                    except Exception as e:
                        logging.error(f"Error calculating {cc.__name__} for {path_to_rtplan_file}: {e}")
                        row.append("N/A")
                
                f.write(", ".join(row) + "\n")

            except Exception as e:
                logging.error(f"Error processing {path_to_rtplan_file}: {e}")
