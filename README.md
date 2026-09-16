# 3DGS-SLAM-TURTLEBOT4

Multi-robot RGB-D SLAM with Gaussian submaps and appearance-aware inter-agent
registration.

## Gaussian-landmark registration in a full SLAM run

The production pipeline no longer needs a `loops.pkl` from an earlier run:

1. each agent builds and saves Gaussian submaps;
2. `LoopDetector` retrieves intra/inter-agent loop candidates from the current
   submaps;
3. candidates are saved to `detected_loops.pkl`;
4. every inter-agent candidate is registered with
   `registration_method: gaussian_landmark`;
5. successful coarse estimates are refined by point-to-plane ICP;
6. the fitness/RMSE filter selects constraints for pose-graph optimization;
7. registered and accepted loops are saved to `loops.pkl` and
   `filtered_loops.pkl` respectively.

Gaussian-landmark registration uses a DINO patch descriptor for appearance,
but replaces the target RGB-D lifted point with the optimized centre of the
corresponding target Gaussian. It is therefore an image-to-Gaussian 3D
registration method, not yet a native descriptor extracted from spherical
harmonics or the other Gaussian parameters.

### Run ReplicaMultiagent

```bash
python run_slam.py configs/ReplicaMultiagent/office_0.yaml \
  --input_path /path/to/ReplicaMultiagent/Office-0 \
  --output_path /path/to/output/office_0_gaussian_landmark \
  --registration_method gaussian_landmark
```

### Run TUM RGB-D

```bash
python run_slam.py \
  configs/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household.yaml \
  --input_path /path/to/TUM_RGBD-SLAM/rgbd_dataset_freiburg3_long_office_household \
  --output_path /path/to/output/tum_gaussian_landmark \
  --registration_method gaussian_landmark
```

FPFH fallback is enabled in the default production configuration. Add
`--disable_fpfh_fallback` for a pure Gaussian-landmark experiment. Use
`--registration_device cuda` only when the main process has enough free GPU
memory for the DINO model and one loaded target Gaussian submap.

The run writes `loop_pipeline_summary.json` together with detailed
`registration_metrics.csv` and `registration_summary.json` diagnostics.
