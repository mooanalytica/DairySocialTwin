# Failed Case: Vision-Language Model (VLM)

This repository contains a minimal failed-case script for classifying cow interaction videos with a vision-language model served through a local OpenAI-compatible vLLM API.

The classifier returns exactly one label:

- `friendly`: grooming or licking.
- `unfriendly`: displacement or headbutting.

## Files

- `classify_cow_interaction_vlm.py`: sends one local video to the configured VLM endpoint and prints the model response.
- `requirements.txt`: lists the Python dependency used directly by the script.

## Runtime Assumptions

The script assumes that an OpenAI-compatible vLLM server is already running at:

```text
http://127.0.0.1:8000/v1
```

The tested endpoint exposes:

```text
Qwen/Qwen3.6-27B-FP8
```

The observed `max_model_len` is `32768`, and the script sets `max_tokens=25000`. The intended hardware environment is `2 x NVIDIA RTX 5090`.

## Configuration

Edit the video path in `classify_cow_interaction_vlm.py` before running:

```python
VIDEO = "file:///videos/D.mp4"
```

Use a `file://` URI that points to the video to be classified. The script does not include video files, generated outputs, or logs.

The API settings are also defined in the script:

```python
client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
    timeout=3600,
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3.6-27B-FP8",
    ...
)
```

## Installation

Install the Python dependency:

```bash
pip install -r requirements.txt
```

## Usage

After the vLLM server is available and `VIDEO` points to a valid local video, run:

```bash
python classify_cow_interaction_vlm.py
```

If vLLM is configured with a reasoning parser, the script may print a separate `REASONING` section before the final `CONTENT` section.

## Repository Hygiene

Large media files, logs, and generated outputs are intentionally not included. If local runs produce logs or regenerated artifacts, they should be kept outside the committed repository or regenerated from the script when needed.
