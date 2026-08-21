# Data Access and Handling

This repository contains source code only. The underlying commercial dairy-farm data and project artifacts are restricted.

Restricted materials include:

- source and derived videos;
- MooAnalytica and Agnovix data;
- human annotations and identity labels;
- Stage 1 tracking, keypoint, identity, and manifest outputs;
- Stage 2 master indexes, split metadata, and feature tables;
- trained checkpoints;
- prediction, evaluation, visualization, and log outputs.

Do not commit, publish, redistribute, or upload these materials to the public repository. Access is limited to authorized project members and must follow the project's data-sharing agreements.

Authorized users can retrieve the staged private assets from:

```text
Yiwen Huang SNA Project Summer 2026/private_assets/stage2_transformer/
```

Restore an asset by removing the `private_assets/stage2_transformer/` prefix and placing it at the resulting relative path in the repository. For example:

```text
private_assets/stage2_transformer/stage2_index/stage2_master_index.csv
-> stage2_index/stage2_master_index.csv
```

The repository does not include a public sample dataset. Do not substitute data from similarly named camera files: resolve every sample to its full video path through the authorized annotation or index metadata, and stop if the mapping is missing or ambiguous.
