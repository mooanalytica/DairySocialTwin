# Project Files, Code Repositories, and Reproducibility Resources

1. The following project resources are stored in the internal Dropbox:

   * The interaction episode dataset described in Section 1.5. The dataset contains 447 source clips at the pre-segmentation level. Of these, 400 source clips are included in the frozen experimental index and were used by the classifiers described in Section 3. The remaining 47 source clips are retained on disk but were excluded from the frozen index.
   * The manually corrected re-identification clips listed in Table 3.
   * The code used to run Stage 1 on the Digital Research Alliance of Canada (DRAC) cluster.
   * Restricted farm data, trained models, full configuration files, and processed results that are not included in the public GitHub repository. This includes, for example, the manifests and model checkpoints for the NRI-style Latent Edge Model.

2. The CSV Segmenter used to split excessively large `.csv` files (Section 2.2) is available at:

   [CSV_SEGMENTER](https://github.com/mooanalytica/DairySocialTwin/tree/main/CSV_SEGMENTER)

3. The dashboard and WebUI, including the social network analysis components (Sections 2.2, 2.4, and 4.1), are available at:

   [DXW](https://github.com/mooanalytica/DairySocialTwin/tree/main/DXW)

4. The directed graph annotation tools (Section 1.1) are available at:

   [Directed_Interaction](https://github.com/mooanalytica/DairySocialTwin/tree/main/Directed_Interaction)

5. The tools for creating camera-specific semantic 2D maps (Section 2.1) are available at:

   * [FloorPlanAnno](https://github.com/mooanalytica/DairySocialTwin/tree/main/FloorPlanAnno)
   * [FloorPlanAnnoSR](https://github.com/mooanalytica/DairySocialTwin/tree/main/FloorPlanAnnoSR)

6. The Interaction Annotator (Section 1.2) is available at:

   [Interaction_Annotator](https://github.com/mooanalytica/DairySocialTwin/tree/main/Interaction_Annotator)

7. The Quality Control tools (Section 1.3) are available at:

   [Quality_Control](https://github.com/mooanalytica/DairySocialTwin/tree/main/Quality_Control)

8. The Trajectory Stabilization tools (Section 1.4) are available at:

   [Trajectory_Stabilization](https://github.com/mooanalytica/DairySocialTwin/tree/main/Trajectory_Stabilization)

9. The re-identification annotation tools (Section 2.3) are available at:

   [re-identification-ANNO](https://github.com/mooanalytica/DairySocialTwin/tree/main/re-identification-ANNO)

10. The re-identification pipeline (Section 4.2) is available at:

    [re-identification](https://github.com/mooanalytica/DairySocialTwin/tree/main/re-identification)

11. The NRI-style Latent Edge Model (Section 3.6) is available at:

    [stage2_NRI](https://github.com/mooanalytica/DairySocialTwin/tree/main/stage2_NRI)

12. The Non-Graph Temporal Model (Section 3.4) is available at:

    [stage2_transformer](https://github.com/mooanalytica/DairySocialTwin/tree/main/stage2_transformer)

13. The unsuccessful Vision-Language Model (VLM) approach discussed in Section 3.1 is available at:

    [vLLM](https://github.com/mooanalytica/DairySocialTwin/tree/main/vLLM)
