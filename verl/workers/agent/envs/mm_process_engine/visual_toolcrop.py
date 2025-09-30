import copy
import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

from verl.workers.agent.tool_envs import ToolBase


class VisualToolCrop(ToolBase):
    name = "visual_toolcrop"

    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        # self.user_prompt = "The image is cropped successfully. Please summarize the model outputs and answer my first question."
        self.user_prompt = "I have marked the region of interest on the image. Please answer the question based on the marked region."
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current image being processed
        
    def extract_answer(self, action_string: str) -> Optional[str]:
        try:
            # pattern = r'<answer>\s*(.*?)\s*</answer>'
            # match = re.search(pattern, action_string)
            match = re.search(r'\{\s*"name"\s*:\s*"Terminate"\s*,\s*"arguments"\s*:\s*\{[^}]*?"answer"\s*:\s*"([^"]+)"', action_string)
            if match:
                return match.group(1)
            return None
        except Exception:
            return None

    def extract_actions(self, text: str):
        """
        Extract only the 'actions' list from the model response text.
        
        Args:
            text (str): The model response text containing actions
            
        Returns:
            Optional[List]: The parsed actions list or None if extraction fails
        """
        try:
            # Try to find the "actions" part using regex
            actions_pattern = r'"actions"\s*:\s*(\[(?:[^\[\]]|\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])*\])'
            # actions_pattern = r'<bbox>\s*(\[(?:[^\[\]]|\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])*\])\s*</bbox>'
            actions_match = re.search(actions_pattern, text)
            
            if not actions_match:
                return None
                
            actions_str = actions_match.group(1)
            actions_list = json.loads(actions_str)
            
            return actions_list


            # parsed_actions = []
            # for action in actions_list:
            #     # If an image (as base64) is provided in the action's arguments, process it.
            #     if image_tool_manager and 'image' in action.get('arguments', {}):
            #         base64_img = action['arguments']['image']
            #         img_key = image_tool_manager.process_base64_image(base64_img)
            #         if img_key:
            #             action['arguments']['image'] = img_key
            #     elif newest_image and 'image' in action.get('arguments', {}):
            #         newest_image_base64 = pil_to_base64(newest_image, url_format=False)
            #         action['arguments']['image'] = newest_image_base64

            #     parsed_actions.append({
            #         "API_name": action["name"],
            #         "API_params": action["arguments"]
            #     })

            
        except Exception as e:
            print(f"Error extracting actions list: {e}")
            return None

    def execute(self, action: dict, **kwargs) -> tuple:
        """
        Execute the crop functionality based on the action dict.
        Args:
            action: The dict containing crop coordinates.
        Returns:
            observation: The structured observation with the processed image.
            reward: 0.5 if crop is successful, 0 otherwise.
            done: Whether the episode is terminated.
            info: Additional info.
        """
        ans = self.extract_answer(action)   
        if ans:
            return "<answer>" + ans + "</answer>", 0.2, True, {"status": "success"}
        try:
            x = self.extract_actions(action)
            # print(f"x: {x}")    
            toolname = x[0].get("name")
            bbox = x[0].get("arguments")["box"]
            # bbox = x
            # print(f"bbox: {bbox}")

            img = self.multi_modal_data['image'][0]
            # img = Image.open("Normal-4-_jpg.rf.79d034045feaccec25cea6f281f40568.jpg") #test
 
            # match = re.match(r'\[\s*([-\d.eE,\s]+)\s*\]', bbox)
            # if match:
            # line_coords = list(map(float, match.group(1).split(',')))
            for line_coords in bbox:
                left, upper, right, lower = line_coords
                if right <= left or lower <= upper:
                    raise ValueError(
                        f"Invalid crop box: right({right}) <= left({left}) or lower({lower}) <= upper({upper})"
                    )
                if not line_coords or len(line_coords) != 4:
                    raise ValueError("Invalid bbox coordinates.")
               # cropped_img = img.crop(line_coords)
                img_with_bbox = img.copy()
                draw = ImageDraw.Draw(img_with_bbox)
                # draw.rectangle([left, upper, right, lower], outline=(255, 0, 0), width=3)
                if img.mode == 'L':  # Grayscale
                    # Use white color for grayscale images
                    draw.rectangle([left, upper, right, lower], outline=255, width=3)
                elif img.mode == 'RGB':
                    # Use RGB color for RGB images
                    draw.rectangle([left, upper, right, lower], outline=(255, 0, 0), width=3)
                elif img.mode == 'RGBA':
                    # Use RGBA color for RGBA images
                    draw.rectangle([left, upper, right, lower], outline=(255, 0, 0, 255), width=3)
                else:
                    # Convert to RGB for other modes
                    img_with_bbox = img_with_bbox.convert('RGB')
                    draw = ImageDraw.Draw(img_with_bbox)
                    draw.rectangle([left, upper, right, lower], outline=(255, 0, 0), width=3)
                img = img_with_bbox
                # width, height = cropped_img.size
                # aspect_ratio = max(width / height, height / width)
                # if aspect_ratio > 200:
                #     raise ValueError(f"absolute aspect ratio must be smaller than 200, got {aspect_ratio}")
            # img.save("test_bbox.jpg")
            obs = {
                "prompt": "<|im_end|>\n<|im_start|>user\n" +"<image>" + self.user_prompt + "<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": [img]}
            }
            reward = 0.2
            done = False
            info = {"status": "success crop"}
            return obs, reward, done, info
            # else:
            #     raise ValueError("Invalid bbox coordinates.")
        except Exception as e:
            obs = "\n<|im_start|>user\n" + f"Error: {str(e)}" + "<|im_end|>\n<|im_start|>assistant\n"
            reward = 0.0
            done = False
            info = {"status": "failed", "error": str(e)}
            return obs, reward, done, info

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'
        
if __name__ == "__main__":
    # Example usage (for testing)
    tool = VisualToolCrop("visual_toolcrop", "Tool for image cropping", {})
    
    # Test crop in tool 
    action = """
    In this image, I initially identified multiple regions that may be related to the gastrointestinal system, such as the structure of the gastrointestinal tract.The coordinate boxes of these areas are [380, 479, 506, 576], [183, 303, 273, 393] and [103, 243, 203, 333].Next, I will use the mark_bbox tool to mark these areas to further confirm whether they are indeed related to the gastrointestinal system. <bbox> [[380, 479, 506, 576], [183, 303, 273, 393], [103, 243, 203, 333]] </bbox>
    """
    obs, reward, done, info = tool.execute(action)
    print(f"Zoom in result - Reward: {reward}, Info: {info}, obs: {obs}")


    action = """
    In this image, I initially identified multiple regions that may be related to the gastrointestinal system, such as the structure of the gastrointestinal tract.The coordinate boxes of these areas are [380, 479, 506, 576], [183, 303, 273, 393] and [103, 243, 203, 333].Next, I will use the mark_bbox tool to mark these areas to further confirm whether they are indeed related to the gastrointestinal system. <bbox> [[380, 479, 506, 576]] </bbox>
    """
    obs, reward, done, info = tool.execute(action)
    print(f"Zoom in result - Reward: {reward}, Info: {info}, obs: {obs}")

    action = """
    In this image, I initially identified multiple regions that may be related to the gastrointestinal system, such as the structure of the gastrointestinal tract.The coordinate boxes of these areas are [380, 479, 506, 576], [183, 303, 273, 393] and [103, 243, 203, 333].Next, I will use the mark_bbox tool to mark these areas to further confirm whether they are indeed related to the gastrointestinal system. <answer> A xxx </answer>
    """
    obs, reward, done, info = tool.execute(action)
    print(f"Zoom in result - Reward: {reward}, Info: {info}, obs: {obs}")
    
    # # Test crop tool 
    # crop_action = """
    # {\"thought\": \"The answer is Yes.\", \"actions\": [{\"name\": \"Terminate\", \"arguments\": {\"ans\": \"NONONO\"}}]}
    # """
    # obs, reward, done, info = tool.execute(crop_action)
    # print(f"Rotate result - Reward: {reward}, Info: {info}, obs: {obs}")
    

    # invalid_action = """
    # {\"thought\": \"The answer is Yes.\", \"actions\": [{\"name\": \"crop\", \"arguments\": {\"ans\": \"NONONO\"}}]}
    # """
    # obs, reward, done, info = tool.execute(invalid_action)
    # print(f"Invalid JSON result - Reward: {reward}, Info: {info}, obs: {obs}")
    
    # tool_action = """
    # {\"thought\": \"To answer this question, I need to focus on the region of the image where the abnormality is present. I will crop the region for further analysis\", \"actions\": [{\"name\": \"crop\", \"arguments\": {\"image\": \"img_1\", \"param\": \"[30.0, 34.0, 466.0, 511.0]\"}}]}"
    # """
    # obs, reward, done, info = tool.execute(tool_action)
    # print(f"tool result - Reward: {reward}, Info: {info}, obs: {obs}")
    

    # tool_action = """
    # {\"thought\": \"To answer this question, I need to focus on the region of the image where the abnormality is present. I will crop the region for further analysis\", \"actions\": [{\"name\": \"crop\", \"arguments\": {\"image\": \"img_1\", \"param\": \"[306.0, 349.0, 46.0, 42.0]\"}}]}"
    # """
    # obs, reward, done, info = tool.execute(tool_action)
    # print(f"tool result - Reward: {reward}, Info: {info}, obs: {obs}")
    
