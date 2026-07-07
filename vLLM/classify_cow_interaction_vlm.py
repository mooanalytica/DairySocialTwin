"""Classify a cow interaction video with a local vLLM OpenAI-compatible API."""

from openai import OpenAI

VIDEO = "file:///videos/D.mp4"

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
    timeout=3600,
)

SYSTEM_PROMPT = """
You are a careful vision-language classifier for cow interactions.
Classify the interaction as exactly one of: friendly or unfriendly.

Friendly = Grooming or Licking.
Unfriendly = Displacement or Headbutting.

Think carefully from visual evidence. Focus on:
1. What contact occurs: mouth/tongue/nose vs head/forehead/body pressure.
2. Whether the contact is gentle/repeated/localized or forceful/sudden.
3. The receiving cow's response: relaxed/stays vs steps back/turns/leaves/yields.
4. Whether any headbutt, shove, displacement, or forced movement happens.

If any clear headbutt, forceful shove, or displacement happens, choose unfriendly.
If the interaction is gentle grooming/licking and accepted, choose friendly.

Return:
Analysis: concise visual reasoning.
Final: friendly OR unfriendly.
"""

resp = client.chat.completions.create(
    model="Qwen/Qwen3.6-27B-FP8",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": VIDEO},
                },
                {
                    "type": "text",
                    "text": "Analyze this video carefully and classify the cow interaction.",
                },
            ],
        },
    ],
    temperature=0.1,
    top_p=0.9,
    max_tokens=25000,  # Keep below the served model's max_model_len and available GPU memory.
    extra_body={
        "top_k": 20,
        "repetition_penalty": 1.05,
        "mm_processor_kwargs": {
            "fps": 8,  # Adjust according to video length, latency, and memory limits.
            "do_sample_frames": True,
        },
        "chat_template_kwargs": {
            "enable_thinking": True,
        },
    },
)

msg = resp.choices[0].message

# When a vLLM reasoning parser is enabled, reasoning may be returned separately.
if hasattr(msg, "reasoning") and msg.reasoning:
    print("REASONING:\n", msg.reasoning)

print("CONTENT:\n", msg.content)
