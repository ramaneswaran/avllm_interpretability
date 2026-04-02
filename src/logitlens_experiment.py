import torch
import csv
from datetime import datetime
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
# Make sure qwen_omni_utils.py is accessible in your environment
from qwen_omni_utils import process_mm_info
from collections import Counter

logit_lens_storage = {}
logit_lens_hooks = []

def logit_lens_hook(layer_idx):
    def hook_fn(module, input, output):
        hidden_state = output[0]
        if hidden_state is not None:
            logit_lens_storage[layer_idx] = hidden_state.detach().cpu()
    return hook_fn

def register_logit_lens_hooks(model):
    global logit_lens_hooks
    for hook in logit_lens_hooks:
        hook.remove()
    logit_lens_hooks.clear()
    logit_lens_storage.clear()
    llm_layers = model.thinker.model.layers
    for layer_idx, layer in enumerate(llm_layers):
        hook = layer.register_forward_hook(logit_lens_hook(layer_idx))
        logit_lens_hooks.append(hook)
    print(f"✅ Registered {len(logit_lens_hooks)} hooks for Logit Lens analysis.")

def clear_logit_lens_hooks():
    global logit_lens_hooks
    for hook in logit_lens_hooks:
        hook.remove()
    logit_lens_hooks.clear()
    print("🧹 Cleared all logit lens hooks.")

def create_token_type_mapping(input_ids, config):
    token_types = []
    for token_id in input_ids.cpu().flatten():
        if token_id == config.audio_token_index:
            token_types.append("audio")
        elif token_id == config.image_token_index:
            token_types.append("image")
        elif token_id == config.video_token_index:
            token_types.append("video")
        else:
            token_types.append("text")
    return token_types

# --- UPDATED ANALYSIS FUNCTION ---
def analyze_and_save_audio_logits_to_csv(model, processor, token_mapping, filename="logit_lens_audio_token_analysis.csv"):
    """
    Processes stored hidden states for audio tokens and saves layer-by-layer predictions to a CSV.
    """
    print(f"\n🔬 Analyzing captured hidden states and saving to '{filename}'...")
    if not logit_lens_storage:
        print("Error: Logit lens storage is empty. No hidden states were captured.")
        return

    # Get the lm_head module. We will NOT move it to CPU.
    # We'll respect its placement by device_map="auto".
    lm_head = model.thinker.lm_head
    lm_head_device = lm_head.weight.device
    print(f"LM Head is on device: {lm_head_device}")

    audio_token_indices = [i for i, t_type in enumerate(token_mapping) if t_type == 'audio']
    if not audio_token_indices:
        print("Warning: No audio tokens found in the input sequence.")
        return

    header = ['Token_Position', 'Token_Type'] + [f'Layer_{i}' for i in sorted(logit_lens_storage.keys())]
    rows = []

    for token_idx in audio_token_indices:
        csv_row = [token_idx, 'audio']
        for layer_idx in sorted(logit_lens_storage.keys()):
            # hidden_state_for_token is currently on CPU
            hidden_state_for_token = logit_lens_storage[layer_idx][0, token_idx, :].unsqueeze(0)

            with torch.no_grad():
                # **THIS IS THE FIX**: Move the CPU tensor to the same device as the lm_head
                logits = lm_head(hidden_state_for_token.to(lm_head_device))

            predicted_token_id = torch.argmax(logits, dim=-1).item()
            predicted_token = processor.tokenizer.decode([predicted_token_id])
            csv_row.append(predicted_token)
        rows.append(csv_row)

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"✅ Successfully saved audio token logit lens analysis to '{filename}'")
    except IOError as e:
        print(f"Error writing to CSV file: {e}")


import argparse

# --- Main Execution Block (Unchanged) ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run logit lens analysis on a video using Qwen2.5 Omni model")
    parser.add_argument("--model_path", required=True, help="Path or name of the pretrained model")
    parser.add_argument("--video_path", required=True, help="Path to the input video file")
    args = parser.parse_args()

    model_path = args.model_path
    video_path = args.video_path

    # --- 1. Model and Processor Setup ---

    print("Loading model...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        attn_implementation="eager",
        device_map="auto"
    )
    model.disable_talker()

    print("Loading processor...")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    # --- 2. Prepare Inputs ---
    conversation = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe what you hear in the video"},
            {"type": "video", "video": video_path},
        ]},
    ]
    USE_AUDIO_IN_VIDEO = True

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    token_mapping = create_token_type_mapping(inputs['input_ids'], model.config.thinker_config)
    print("Token mapping created:", Counter(token_mapping))

    # --- 3. Perform Analysis via Direct Forward Pass ---
    print("\nRunning a direct forward pass for logit lens analysis...")
    register_logit_lens_hooks(model)

    with torch.no_grad():
        outputs = model.thinker(**inputs, output_hidden_states=True)

    analyze_and_save_audio_logits_to_csv(model, processor, token_mapping)
    clear_logit_lens_hooks()

    # --- 4. (Optional) Generate Text Output to Confirm Model Works ---
    print("\nRunning model.generate() to get the final text output...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)

    print("\n--- Generated Text ---")
    decoded_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(decoded_text[0])
    print("\nAnalysis complete.")