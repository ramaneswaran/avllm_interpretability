import torch
import pickle
from datetime import datetime
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Tuple
from contextlib import contextmanager

# Ensure the model's source files are accessible
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import argparse
import os
import time
import json

def format_time(seconds):
    """Convert seconds to a readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# --- 1. Type Mapping ---
# Define a consistent numeric mapping for token types
TOKEN_TYPE_MAP: Dict[str, int] = {
    "query_text": 0,
    "audio": 1,
    "video": 2,
    "image": 3,
    "generated": 4,
}
# Create a reverse map for potential debugging (optional)
ID_TO_TOKEN_TYPE: Dict[int, str] = {v: k for k, v in TOKEN_TYPE_MAP.items()}


# --- 2. The (Vectorized) Hook for Modifying Attention Masks ---
class BlockAttentionHook:
    """
    A forward pre-hook that dynamically modifies the attention mask for
    BOTH prefill (prompt processing) and generation (token-by-token).
    """
    def __init__(self, 
                 numeric_knockout_rules: torch.Tensor, 
                 numeric_token_types: torch.Tensor, 
                 original_input_len: int,
                 generated_token_id: int):
        
        self.numeric_knockout_rules = numeric_knockout_rules
        self.numeric_token_types = numeric_token_types
        self.original_input_len = original_input_len
        self.generated_token_id = generated_token_id
        
        # Optimization: only run full logic if a query token is a 'source'
        self.source_types_to_block = torch.unique(self.numeric_knockout_rules[:, 0])
        self.current_device = None # Set on first call

    def __call__(self, module, args, kwargs):
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None or self.numeric_knockout_rules.shape[0] == 0:
            return args, kwargs

        if self.current_device is None:
            self.current_device = attention_mask.device

        k_len = attention_mask.shape[-1]
        q_len = attention_mask.shape[-2]
        
        modified_mask = None
        mask_value = torch.finfo(attention_mask.dtype).min
        
        # --- CASE 1: Generation (Autoregressive step) ---
        # Query shape is [1, K_len], where K_len is dynamic
        if q_len == 1:
            query_pos = k_len - 1 

            # 1. Determine query type
            if query_pos >= self.original_input_len:
                query_type_id = self.generated_token_id
            else:
                # This can happen in prompts with length > 1
                query_type_id = self.numeric_token_types[query_pos].item()

            # 2. OPTIMIZATION
            if query_type_id not in self.source_types_to_block:
                return args, kwargs

            # 3. Determine key types
            num_generated = k_len - self.original_input_len
            if num_generated > 0:
                generated_ids = torch.full(
                    (num_generated,), 
                    self.generated_token_id, 
                    dtype=self.numeric_token_types.dtype, 
                    device=self.numeric_token_types.device 
                )
                key_type_ids = torch.cat((self.numeric_token_types, generated_ids))
            else:
                # We are attending to a prefix of the original input
                key_type_ids = self.numeric_token_types[:k_len]

            # 4. Build block mask (1D)
            final_block_mask = torch.zeros(k_len, dtype=torch.bool, device=self.current_device)
            for rule_idx in range(self.numeric_knockout_rules.shape[0]):
                source_id, target_id = self.numeric_knockout_rules[rule_idx]
                if query_type_id == source_id:
                    key_is_target_mask = (key_type_ids == target_id)
                    final_block_mask.logical_or_(key_is_target_mask)
            
            # 5. Apply (1D)
            if torch.any(final_block_mask):
                modified_mask = attention_mask.clone()
                modified_mask[..., 0, final_block_mask] = mask_value

        # --- CASE 2: Prefill (Prompt processing) ---
        # Query and Key are both the original input
        elif q_len == k_len and k_len == self.original_input_len:
            query_type_ids = self.numeric_token_types
            key_type_ids = self.numeric_token_types

            # Build block mask (2D)
            final_block_mask = torch.zeros((q_len, k_len), dtype=torch.bool, device=self.current_device)
            
            for rule_idx in range(self.numeric_knockout_rules.shape[0]):
                source_id = self.numeric_knockout_rules[rule_idx, 0]
                target_id = self.numeric_knockout_rules[rule_idx, 1]

                # Find all queries of source type
                query_is_source_mask = (query_type_ids == source_id)
                # Find all keys of target type
                key_is_target_mask = (key_type_ids == target_id)
                
                # Create a 2D mask for (Q=src, K=tgt)
                # [Q_len, 1] & [1, K_len] -> [Q_len, K_len]
                q_k_mask = query_is_source_mask.unsqueeze(1) & key_is_target_mask.unsqueeze(0)
                final_block_mask.logical_or_(q_k_mask)

            # 5. Apply (2D)
            if torch.any(final_block_mask):
                modified_mask = attention_mask.clone()
                # Apply to all batches/heads
                modified_mask[..., final_block_mask] = mask_value
        
        if modified_mask is not None:
            kwargs["attention_mask"] = modified_mask
        
        return args, kwargs

# --- 3. The Attention *Capture* Hook ---

def attention_hook_fn(layer_idx: int, storage_dict: dict):
    """
    Creates a hook function that captures attention weights and
    stores them in the provided storage_dict.
    """
    def hook(module, input, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            storage_dict[layer_idx].append(output[1].detach().cpu())
    return hook


# --- 4. The Context Manager to Apply All Hooks ---

@contextmanager
def block_attention(model: torch.nn.Module, 
                    knockout_rules: List[Tuple[str, str, int, int]], # <-- NEW FORMAT
                    token_types: List[str], 
                    original_input_len: int,
                    track_attention: bool = False,
                    capture_layer_range: Optional[Tuple[int, int]] = None): # <-- MODIFIED
    """
    A context manager that applies attention hooks to the model.
    - knockout_rules: List of (source, target, start_layer, end_layer)
    - capture_layer_range: Optional (start, end) range for *attention capture*
    """
    
    knockout_hook_handles = []
    capture_hook_handles = []
    attention_storage = defaultdict(list)
    
    # 1. Pre-process token types
    try:
        device = model.device 
        
        numeric_token_types = torch.tensor(
            [TOKEN_TYPE_MAP[t] for t in token_types], dtype=torch.long, device=device
        )
        generated_token_id = TOKEN_TYPE_MAP["generated"]

    except KeyError as e:
        print(f"Error: Invalid token type or rule name: {e}")
        print(f"Valid types are: {list(TOKEN_TYPE_MAP.keys())}")
        yield attention_storage
        return
    
    try:
        # 2. Register all hooks
        num_knockout_hooks = 0
        num_capture_hooks = 0
        
        all_layers = model.thinker.model.layers
        total_layers = len(all_layers)
        
        # --- Define the start/end for attention CAPTURE ---
        capture_start, capture_end = 0, total_layers
        if capture_layer_range:
            capture_start = max(0, capture_layer_range[0])
            capture_end = min(total_layers, capture_layer_range[1])
            print(f"Applying ATTENTION CAPTURE to layer range: [{capture_start}, {capture_end})")
        
        # --- NEW: Layer-by-layer hook registration ---
        for i, layer in enumerate(all_layers):
            
            # --- Part 1: Register Knockout Hook ---
            
            # Find rules active for this layer `i`
            active_rules_for_layer = []
            for rule in knockout_rules:
                if len(rule) != 4:
                    raise ValueError(f"Rule {rule} is not in the format (src, tgt, start_layer, end_layer)")
                src, tgt, start, end = rule
                if start <= i < end:
                    active_rules_for_layer.append((src, tgt))
            
            # Determine layer device
            try:
                layer_device = next(layer.parameters()).device
            except StopIteration:
                try:
                    layer_device = next(layer.buffers()).device
                except StopIteration:
                    layer_device = model.device 
            
            hook_token_types = numeric_token_types.to(layer_device)

            # Register the knockout (pre-)hook if there are active rules
            if active_rules_for_layer:
                numeric_rules_list = [
                    (TOKEN_TYPE_MAP[src], TOKEN_TYPE_MAP[tgt]) for src, tgt in active_rules_for_layer
                ]
                numeric_knockout_rules = torch.tensor(
                    numeric_rules_list, dtype=torch.long, device=layer_device
                )
                
                knockout_hook = BlockAttentionHook(
                    numeric_knockout_rules=numeric_knockout_rules,
                    numeric_token_types=hook_token_types,
                    original_input_len=original_input_len,
                    generated_token_id=generated_token_id
                )
                handle = layer.self_attn.register_forward_pre_hook(knockout_hook, with_kwargs=True)
                knockout_hook_handles.append(handle)
                num_knockout_hooks += 1

            # --- Part 2: Register Capture Hook ---
            if track_attention and (capture_start <= i < capture_end):
                capture_hook = attention_hook_fn(i, attention_storage)
                handle = layer.self_attn.register_forward_hook(capture_hook)
                capture_hook_handles.append(handle)
                num_capture_hooks += 1
        
        if knockout_hook_handles:
            print(f"Applied {num_knockout_hooks} total knockout hooks across layers.")
            print(f"Rule configuration: {knockout_rules}")
        if capture_hook_handles:
            print(f"Applied {num_capture_hooks} attention capture hooks.")
            
        # 3. Yield control back to the `with` block
        yield attention_storage
    
    finally:
        # 4. Clean up all hooks
        for handle in knockout_hook_handles:
            handle.remove()
        for handle in capture_hook_handles:
            handle.remove()
        
        if knockout_hook_handles:
            print(f"Removed {num_knockout_hooks} knockout hooks.")
        if capture_hook_handles:
            print(f"Removed {num_capture_hooks} capture hooks.")


# --- 5. Data Capture and Saving ---

def save_knockout_data(filename: str, 
                       generated_text: str, 
                       token_map: List[str], 
                       knockout_rules: List[Tuple[str, str, int, int]], # <-- NEW FORMAT
                       attention_storage: dict,
                       capture_layer_range: Optional[Tuple[int, int]]): # <-- NEW
    """
    Saves the results of the knockout experiment to a pickle file.
    """
    data = {
        'attention_weights_per_step': attention_storage,
        'token_mapping': token_map,
        'generated_text': generated_text,
        'metadata': {
            'knockout_rules': knockout_rules,
            'capture_layer_range': capture_layer_range,
            'model_name': model.config.name_or_path,
            'original_input_length': len(token_map),
            'num_layers_tracked': len(attention_storage) if attention_storage else 0,
            'timestamp': datetime.now().isoformat(),
        }
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Saved knockout experiment data to {filename}")


def create_token_type_mapping(input_ids: torch.Tensor, config) -> List[str]:
    """
    Maps each input ID to its modality type (str).
    """
    token_types = []
    # Assumes batch size is 1
    for token_id in input_ids[0]:
        tid = token_id.item()
        if tid == config.audio_token_index: token_types.append("audio")
        elif tid == config.image_token_index: token_types.append("image") 
        elif tid == config.video_token_index: token_types.append("video")
        else: token_types.append("query_text")
    return token_types

# --- 6. Main Execution Block ---

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--knockout-rules',
        type=str,
        default='generated,video,0,35',
        help='Knockout rules as semicolon-separated pairs. Example: "generated,video,0,35;audio,text"'
    )
    
    parser.add_argument(
        '--track-attention',
        action='store_true',
        help='Enable attention tracking'
    )

    parser.add_argument(
        '--model_path',
        required=True,
        help='Path to the pretrained model directory (e.g., Qwen/Qwen2.5-Omni-3B)'
    )

    parser.add_argument(
        '--video_path',
        required=True,
        help='Path to the input video file used for evaluation (e.g., assets/02321.mp4)'
    )

    parser.add_argument(
        '--query_text',
        default='Describe what you see',
        help='Text query to prompt the model (default: "Describe what you see")'
    )

    args = parser.parse_args()
    
    # Parse knockout rules
    knockout_rules = []
    for pair in args.knockout_rules.split(';'):
        src_tok_type, tgt_tok_type, start_layer, end_layer = pair.strip().split(',')
        knockout_rules.append((src_tok_type.strip(), tgt_tok_type.strip(), int(start_layer.strip()), int(end_layer.strip())))
    print(knockout_rules)
    
    return {
        'KNOCKOUT_RULES': knockout_rules,
        'TRACK_ATTENTION': args.track_attention,
        'MODEL_PATH': args.model_path,
        'VIDEO_PATH': args.video_path,
        'QUERY_TEXT': args.query_text,
    }


if __name__ == "__main__":

    config = parse_args()
    KNOCKOUT_RULES = config['KNOCKOUT_RULES']
    TRACK_ATTENTION = False
    MODEL_PATH = config['MODEL_PATH']
    VIDEO_PATH = config['VIDEO_PATH']
    QUERY_TEXT = config['QUERY_TEXT']

    # This range is now ONLY for attention capture
    CAPTURE_LAYER_RANGE = (0, 10)

    rules_str = "_".join([f"{s}2{t}L{start}_{end}" for s, t, start, end in KNOCKOUT_RULES]) if KNOCKOUT_RULES else "baseline"
    save_path = f"{rules_str}_single_video.json"

    print("🚀 Loading model and processor...")

    print("Loading original model...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype="auto", 
        attn_implementation="eager", # Eager is needed for hooks
        device_map="auto"
    )
    model.disable_talker()
    
    print("Loading processor...")

    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    
    # You can get the total number of layers to help define rules:
    total_layers = model.thinker.model.config.num_hidden_layers
    print(f"Model has {total_layers} layers.")

    total_time = 0
    sample_idx = 0
    print("-"*10)
    print(f"Processing single video: {VIDEO_PATH}")

    start_time = time.time()
    video_path = VIDEO_PATH

    conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUERY_TEXT},
                    {"type": "video", "video": video_path},
                ],
            },
        ]

    USE_AUDIO_IN_VIDEO = True

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    token_mapping = create_token_type_mapping(inputs['input_ids'], model.config.thinker_config)
    original_input_len = len(token_mapping)
    print(f"Original input length: {original_input_len}")
    print(f"Token counts: {Counter(token_mapping)}")

    with block_attention(model, 
                        KNOCKOUT_RULES, 
                        token_mapping, 
                        original_input_len, 
                        track_attention=TRACK_ATTENTION,
                        capture_layer_range=CAPTURE_LAYER_RANGE # <-- MODIFIED
                        ) as attention_storage:
    
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=50,
            output_attentions=TRACK_ATTENTION,
            return_dict_in_generate=True,
        ).sequences

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    print("\n--- Results ---")
    print(f"Video Path: {VIDEO_PATH}")
    print(f"Knockout Rules: {KNOCKOUT_RULES}")
    print(f"Capture Layer Range: {CAPTURE_LAYER_RANGE}")
    print(f"Token Mapping Length: {len(token_mapping)}")
    print(f"Generated Text:\n{generated_text}")
