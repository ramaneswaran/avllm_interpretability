# Based on https://aclanthology.org/2023.emnlp-main.841.pdf

from openai import OpenAI
import time
import concurrent.futures
import json
from openai import OpenAI, APIError
from typing import List, Dict, Any, Optional
import re 
from datetime import timedelta
import gc
from pathlib import Path 
import argparse 


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio-Visual Language Model Interpretability Evaluation"
    )
    
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input JSON file")

    parser.add_argument("--metadata_path", type=str, required=True,
                        help="Path to metadata JSON file")
    
    parser.add_argument("--modality", type=str, choices=["audio", "video"], required=True,
                        help="Modality to evaluate (audio or video)")
    parser.add_argument("--max_workers", type=int, default=16,
                        help="Number of concurrent workers")
    
    return parser.parse_args()

class Evaluator:
    def __init__(self, api_key: str, api_base_url: str, model_name: str):
        """
        Initializes the Evaluator with API credentials and the model name.

        Args:
            api_key (str): The API key for the LLM service.
            api_base_url (str): The base URL of the API endpoint (e.g., "http://localhost:8000/v1").
            model_name (str): The identifier of the model to be used for evaluations.
        """
        if not api_key or not api_base_url or not model_name:
            raise ValueError("API key, base URL, and model name cannot be empty.")

        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base_url,
        )

    def _apply_audio_prompt_template(self, item) -> str:
        """
        Apply prompt template to a data item.
        """
        prompt_template =  """Evaluate how accurately the candidate caption describes the audio compared to the reference caption.
                            The caption is an audio-visual caption and may include visual descriptions. This is fine and should be ignored
                            Specifically, you should neither reward nor penalize candidate caption for any visual description and focus only on audio-related content
                            On the other hand, you should penalize model if it hallucinates some audio events

                        Candidate Caption: I see a grassy area with a few pigeons. There are some rocks and a wooden fence in the background. I hear some birds chirping and a bit of rustling. There's also a sound of a drill in the background, and then a sigh. What do you think about this scene?
                        Reference Caption: drilling with music playing and a man speaks
                        score: 0.25
                        reason: The candidate correctly identifies the drilling sound but misses two major audio events (music playing and man speaking). More critically, it hallucinates three audio events that don't exist in the reference (birds chirping, rustling, and a sigh), suggesting significant audio misperception or confabulation.
                        
                        Candidate Caption: Well, in the video, I see a baby. The baby is being held by someone. The baby looks a bit chubby and has a small head. The baby's skin is smooth and soft. The baby's eyes are open, and it seems to be looking at the person holding it. The baby's mouth is open, and it looks like it might be making a sound or crying. Oh, and there's a bit of a smile on the person's face who's holding the baby. There's also some background noise, but it's not really clear what it is. It could be some kind of ambient sound in the room. So, what do you think about the video? Do you have any other questions?
                        Reference Caption: a man laughs followed by a baby crying
                        score: 0.5
                        reason: The candidate correctly identifies baby sounds (mentions baby "making a sound or crying"), capturing one of the two main audio events. However, it completely misses the man laughing, which is a distinct and salient audio event in the reference. The vague mention of "background noise" and "ambient sound" shows audio awareness but lacks the specificity of identifying laughter.

                        Candidate Caption: Well, in the video, you can see the inside of a car. There's a red dashboard with some stickers on it. You can also see the steering wheel and a few other bits of the car's interior. As for what I hear, there's a constant hum of the engine, which is probably from the car being in motion. And there's also the sound of the wind outside, like it's a nice day. So, what do you think about the video? Do you like the sound of the car engine?
                        Reference Caption: a vehicle engine running smoothly
                        score: 0.80
                        reason: The candidate accurately identifies the primary audio event - the vehicle engine running (described as "constant hum of the engine"). However, it hallucinates an additional audio event by mentioning "the sound of the wind outside," which is not present in the reference caption. The core audio description is correct, but the added wind sound warrants a moderate penalty.

                        Candidate Caption: Well, in the video, you can see a close - up of a car engine. There are all kinds of parts like the air filter, the belts, and the hoses. The engine looks pretty complex with all these different components. And, um, there's a sound of the engine starting up. It's a bit of a loud, rumbling noise. What do you think about the engine? Do you have any questions about it?
                        Reference Caption: an engine revs repeatedly
                        score: 0.70 
                        reason: The candidate correctly identifies the engine sound as the primary audio event with appropriate descriptors ("loud, rumbling noise"). While it describes this as "starting up" rather than "revs repeatedly," it still captures the essential audio content - a loud engine sound. The missing detail about the repetitive revving pattern is a relatively minor omission compared to getting the core audio event correct.

                        Candidate Caption: <CANDIDATE_CAPTION>
                        Reference Caption: <REFERENCE_CAPTION>

                        Rate the accuracy precisely in 0-1 scale.
                        You need to strictly return your output as a json with the following keys
                        - score: A precise value between 0 and 1
                        - reason: Reasoning for why you provided that score
                        """
        # return prompt_template.format(input_text)
        reference_captions = item['audio_captions']

        return prompt_template.replace("<CANDIDATE_CAPTION>", item['output_text']).replace("<REFERENCE_CAPTION>", reference_captions[0])


    def _apply_vision_prompt_template(self, item) -> str:
        """
        Apply prompt template to a data item.
        """

        prompt_template =  """Evaluate how accurately the candidate caption describes the video compared to the reference caption.
                            The caption is an audio-visual caption and may include audio descriptions. This is fine and should be ignored
                            Specifically, you should neither reward nor penalize candidate caption for any audio description and focus only on vision-related content
                            On the other hand, you should penalize model if it hallucinates some visual events
                            Lastly, we are not expecting very detailed caption, so even an accurate but high-level description of key details is acceptable.

                        Candidate Caption: Well, in the video, I see a bunch of pigeons. There's one pigeon walking on the grass first. Then, there are more pigeons in the background, some are flying around. The grass is green and a bit dry in some places. There are also some trees and a building in the background. Oh, and there's a bird cooing in the background too. It's a pretty peaceful scene. So, what do you think about it? Do you have any other questions about the video?
                        Reference Caption: drilling with music playing and a man speaks
                        score: 0.9
                        reason: The candidate accurately captures the high-level visual scene (multiple pigeons, grass, outdoor space with trees/building, some flying). The "pigeon walking on the grass" likely refers to the spinning/rolling behavior. Minor details like snow patches and pigeon color variation are omitted but don't significantly impact accuracy. Audio description (cooing) is appropriately ignored per instructions.
                        
                        Candidate Caption: Well, in the video, I see a baby. The baby is being held by someone. The baby looks a bit chubby and has a small head. The baby's skin is smooth and soft. The baby's eyes are open, and it seems to be looking at the person holding it. The baby's mouth is open, and it looks like it might be making a sound or crying. Oh, and there's a bit of a smile on the person's face who's holding the baby. There's also some background noise, but it's not really clear what it is. It could be some kind of ambient sound in the room. So, what do you think about the video? Do you have any other questions?
                        Reference Caption: A baby is being held upright and their initial expression appears neutral or happy while looking at an adult facing them. Over the sequence, the baby's facial expression gradually shifts to one of distress, and by the end, the baby looks like they are crying. The adults face is visible near the baby throughout, possibly interacting with them.
                        score: 0.8
                        reason: The candidate accurately captures the main visual elements: baby being held, looking at the person holding them, mouth open (suggesting crying), and the adult's visible face/expression. While it doesn't explicitly describe the temporal progression from neutral to distressed, it does capture the crying state which is the endpoint. The core visual content is largely correct.

                        Candidate Caption: Well, in the video, I see a man in a black shirt and beige pants walking through a room. There's a computer on a desk, a clock on the wall, and a bulletin board with some papers on it. The room has a checkered floor. As for the sound, there's a baby crying in the background. It's a bit loud and really makes you feel the atmosphere of the room. So, what do you think about the video? Do you have any other questions?
                        Reference Caption: A man enters an office-like room, walks towards the camera, then suddenly appears to get startled or react in surprise near the foreground. He lifts his arms above his head and then quickly turns and looks around, suggesting he may have seen or experienced something unexpected in the room.
                        score: 0.6
                        reason: The candidate accurately describes the man, his clothing, and the room setting (desk, computer, clock, bulletin board, checkered floor), capturing the environmental context well. It identifies him walking through the room. However, it misses the startled reaction, arm-raising, and surprised behavior - the key dramatic action. Still, it gets roughly half the visual content (person, setting, basic movement) correct.

                        Candidate Caption: In the image, I see a person standing in a room with a white wall and a window in the background. The person is wearing a white shirt and appears to be looking at something off to the side. The room is well-lit, and there is a piece of furniture, possibly a desk or a table, in the foreground. The overall atmosphere of the room is calm and quiet.
                        Reference Caption: A man enters an office-like room, walks towards the camera, then suddenly appears to get startled or react in surprise near the foreground. He lifts his arms above his head and then quickly turns and looks around, suggesting he may have seen or experienced something unexpected in the room.
                        score: 0.0 
                        reason: The candidate describes a static, calm scene (person standing, looking to side, well-lit room) while the reference describes dynamic action (man entering, walking, getting startled, lifting arms, turning). The candidate completely misses all the key motion and dramatic reaction events. It captures only the most basic element (a person in a room) but fails to convey any of the actual visual narrative or action.

                        Candidate Caption: <CANDIDATE_CAPTION>
                        Reference Caption: <REFERENCE_CAPTION>

                        Rate the accuracy precisely in 0-1 scale.
                        You need to strictly return your output as a json with the following keys
                        - score: A precise value between 0 and 1
                        - reason: Reasoning for why you provided that score
                        """
        reference_captions = item['video_captions']
        return prompt_template.replace("<CANDIDATE_CAPTION>", item['output_text']).replace("<REFERENCE_CAPTION>", reference_captions[0])

    def __call__(self, data, modality: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Main entry point for the evaluator. Orchestrates the entire evaluation pipeline.

        Args:
            json_path (str): The path to the input JSON dataset file.
            **kwargs: Additional keyword arguments to be passed to the LLM API calls.

        Returns:
            List[Dict[str, Any]]: The processed dataset with LLM answers and sanitized outputs.
        """
        print(f"--- Starting Evaluation Pipeline ---")

        dataset = self._prepare_dataset(data, modality)

        result_dataset = self._run_llm_inference(dataset, **kwargs)

        final_outputs = self._sanitize_outputs(result_dataset)

        return final_outputs

    def _prepare_dataset(self, data: str, modality = 'audio') -> List[Dict[str, Any]]:
        
        # Apply prompt template to each item
        processed_data = []
        for item in data:

            if 'output_text' not in item:
                continue
            if modality == 'audio':
                prompt = self._apply_audio_prompt_template(item)
            else:
                prompt = self._apply_vision_prompt_template(item)
            item['prompt'] = prompt
            processed_data.append(item)
        
        
        return processed_data

    def _run_llm_inference(
        self,
        dataset: List[Dict[str, Any]],
        max_workers: int = 5,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Runs LLM inference in parallel on a dataset. It reads the 'prompt' key
        from each dictionary, calls the LLM, and adds the response back into
        the same dictionary under the key 'llm_answer'.

        Args:
            dataset (List[Dict[str, Any]]): A list of dictionaries, each with a 'prompt'.
            max_workers (int): Maximum number of concurrent threads.
            **kwargs: Additional keyword arguments for the API calls.

        Returns:
            List[Dict[str, Any]]: The dataset with the 'llm_answer' key added to each dictionary.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(
                    self._make_request_with_retry,
                    messages=[{"role": "user", "content": item.get("prompt", "")}],
                    **kwargs
                ): item
                for item in dataset if item.get("prompt")
            }

            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                prompt_snippet = item.get("prompt", "")
                try:
                    response = future.result()
                    item['answer'] = response
                except Exception as exc:
                    item['answer'] = None
        return dataset

    def _sanitize_outputs(self, outputs):

        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'

        for output in outputs:

            output.pop('prompt')
            full_output = output['answer'].get('choices', [{}])[0].get('message', {}).get('content', None)
            
            if full_output is None:
                full_output = "No output"

            post_think_output = full_output.split('</think>')[-1]
            match = re.search(pattern, post_think_output)
            if match:
                json_str = match.group()
                try:
                    data = json.loads(json_str)
                except Exception as error:
                    data = {'score': 0.0, 'reason': f'[ERROR] {error}'}
                output['answer'] = data 
            else:
                output['answer'] = {'score': 0.0, 'reason': f'[NO MATCH]'}

        return outputs

    def _make_request_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        delay: int = 2,
        **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Internal method to make a single API call with an exponential backoff retry strategy.
        """
        for attempt in range(max_retries):
            try:
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 8192),
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.8),
                    "presence_penalty": kwargs.get("presence_penalty", 1.5),
                    "extra_body": kwargs.get("extra_body", {
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": True},
                    }),
                }
                params.update(kwargs)
                chat_response = self.client.chat.completions.create(**params)
                return chat_response.model_dump()
            except APIError as e:
                print(f"API Error on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {delay}s...")
            except Exception as e:
                print(f"An unexpected error occurred on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {delay}s...")

            if attempt + 1 < max_retries:
                time.sleep(delay)
                delay *= 2
            else:
                print(f"Max retries reached for prompt: {messages[0]['content'][:50]}... Request failed.")
                return None

def read_json(input_path):
    with open(input_path, 'r') as fin:
        input_data = json.load(fin)
    return input_data 

if __name__ == "__main__":

    OPENAI_API_KEY = "token-abc123"
    OPENAI_API_BASE = "http://localhost:8000/v1"
    MODEL_NAME = "Qwen/Qwen3-32B"

    evaluator = Evaluator(
        api_key=OPENAI_API_KEY,
        api_base_url=OPENAI_API_BASE,
        model_name=MODEL_NAME
    )


    args = parse_args()

    meta_data = read_json(args.metadata_path)
    input_data = read_json(args.input_path)
    output_dir = Path('evaluated_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = output_dir / f"{Path(args.input_path).stem}_{args.modality}.json"

    
    # Add metadata
    if len(meta_data) == len(input_data):
        for idx in range(len(input_data)):
            
            if 'id' not in input_data[idx]:
                continue

            assert input_data[idx]['id'] == meta_data[idx]['id']
            input_data[idx]['video_captions'] = meta_data[idx]['video_captions']

    start_time = time.time()
    final_results = evaluator(data=input_data, modality=args.modality, max_workers=16)
    # OUTPUT_PATH = "exp_gpt.json"
    with open(OUTPUT_PATH, 'w') as fout:
        json.dump(final_results, fout, indent=2)
    end_time = time.time()
    print("\n------------------")
    print(f"Total time elapsed: {str(timedelta(seconds=int(end_time - start_time)))}")
