#!/bin/bash
export VLLM_USE_TRTLLM_ATTENTION=1 
export VLLM_USE_TRTLLM_DECODE_ATTENTION=1 
export VLLM_USE_TRTLLM_CONTEXT_ATTENTION=1 
export VLLM_USE_FLASHINFER_MXFP4_BF16_MOE=1


CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "Qwen/Qwen3-VL-32B-Instruct" --dtype auto --api-key token-abc123 --port 8000 --tensor-parallel-size 4
