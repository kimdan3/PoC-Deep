"""
DeepSeek API Client for generating marketing insights
"""

import os
import logging
import httpx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Chat message structure"""
    role: str
    content: str

@dataclass
class ChatCompletionResponse:
    """Chat completion response structure"""
    choices: List[Dict[str, Any]]

class DeepSeekClient:
    """Client for interacting with the DeepSeek API"""
    
    BASE_URL = "https://api.deepseek.com/v1/chat/completions"
    
    def __init__(self):
        """Initialize the DeepSeek client"""
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
        
        self.client = httpx.Client(
            timeout=float(os.getenv('TIMEOUT_SECONDS', 90)),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> ChatCompletionResponse:
        """
        Generate chat completion using DeepSeek API
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            ChatCompletionResponse: API response
            
        Raises:
            Exception: If the API request fails
        """
        try:
            response = self.client.post(
                self.BASE_URL,
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Log response structure
            logger.debug(f"DeepSeek API response structure: {data}")
            
            # Safely parse response
            choices = []
            for choice in data.get("choices", []):
                content = ""
                if isinstance(choice, dict):
                    # If message object exists
                    if "message" in choice and isinstance(choice["message"], dict):
                        content = choice["message"].get("content", "")
                    # If no message object
                    elif "text" in choice:
                        content = choice["text"]
                    # If content exists directly
                    elif "content" in choice:
                        content = choice["content"]
                
                choices.append({
                    "message": {
                        "content": content
                    }
                })
            
            return ChatCompletionResponse(choices=choices)
            
        except httpx.HTTPStatusError as e:
            logger.error(f"DeepSeek API request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during DeepSeek API request: {str(e)}")
            raise 