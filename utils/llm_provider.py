"""
LLM Provider Abstraction Layer

Supports multiple LLM providers (Ollama, Gemini) with a unified interface.
Provider selection is controlled via config.yaml.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import json


class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Generate a chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            top_p: Nucleus sampling parameter (0-1), None to use default
            
        Returns:
            Generated text response
        """
        pass
    
    def chat_completion_with_logprobs(
        self,
        messages: List[Dict[str, str]],
        possible_labels: List[str],
        temperature: float = 0.7,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate a chat completion with log probabilities for classification.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            possible_labels: List of possible label strings to get probabilities for
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            
        Returns:
            Dict with 'prediction' (str) and 'probabilities' (Dict[str, float])
            If not supported, returns None for probabilities
        """
        # Default implementation: not supported
        # Providers that support logprobs should override this
        prediction = self.chat_completion(messages, temperature, max_tokens, stop)
        return {
            'prediction': prediction,
            'probabilities': None  # Not supported
        }
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider (local models)"""
    
    def __init__(self, config: dict):
        """
        Initialize Ollama provider.
        
        Args:
            config: Ollama configuration from config.yaml
        """
        try:
            import ollama
            self.client = ollama.Client(host=config.get('base_url', 'http://localhost:11434'))
            self.model = config.get('model', 'qwen2.5:7b')
            print(f"INFO: Initialized Ollama provider with model: {self.model}")
        except ImportError:
            raise ImportError(
                "Ollama package not installed. Install with: pip install ollama"
            )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
        top_p: Optional[float] = None
    ) -> str:
        """Generate chat completion using Ollama"""
        try:
            options = {
                'temperature': temperature,
                'num_predict': max_tokens,
            }
            
            if stop:
                options['stop'] = stop
            
            if top_p is not None:
                options['top_p'] = top_p
            
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=options
            )
            
            return response['message']['content']
            
        except Exception as e:
            print(f"ERROR: Ollama API call failed: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for Ollama models.
        Uses simple word-based estimation (roughly 1.3 tokens per word).
        """
        words = len(text.split())
        return int(words * 1.3)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider (supports both standard OpenAI and Azure OpenAI)"""
    
    def __init__(self, config: dict):
        """
        Initialize OpenAI provider.
        
        Args:
            config: OpenAI configuration from config.yaml
        """
        try:
            import tiktoken
            
            api_key = config.get('api_key')
            if not api_key or api_key == 'YOUR_AZURE_OPENAI_API_KEY_HERE' or api_key == 'YOUR_OPENAI_API_KEY_HERE':
                raise ValueError(
                    "OpenAI API key not configured. Add your key to config.yaml"
                )
            
            # Support for Azure OpenAI endpoint
            azure_endpoint = config.get('azure_endpoint')
            if azure_endpoint:
                # Azure OpenAI configuration
                from openai import AzureOpenAI
                
                # Azure endpoint should be base URL without /openai suffix
                base_endpoint = azure_endpoint.replace('/openai', '')
                
                self.client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=base_endpoint,
                    api_version="2024-08-01-preview"  # Use latest API version
                )
                print(f"INFO: Initialized Azure OpenAI provider with endpoint: {base_endpoint}")
            else:
                # Standard OpenAI configuration
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                print(f"INFO: Initialized OpenAI provider")
            
            self.model = config.get('model', 'gpt-3.5-turbo')
            self.is_azure = azure_endpoint is not None
            
            # Use tiktoken for token counting if available
            try:
                self.encoding = tiktoken.encoding_for_model("gpt-4")  # Use gpt-4 encoding as fallback
            except:
                # For custom models (like gpt-5-nano), use cl100k_base encoding
                self.encoding = tiktoken.get_encoding("cl100k_base")
            
            print(f"INFO: Using model: {self.model}")
            
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai tiktoken"
            )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
        top_p: Optional[float] = None
    ) -> str:
        """Generate chat completion using OpenAI"""
        try:
            # GPT-5 reasoning models (o1, o3) use max_completion_tokens and reasoning_effort
            if "gpt-5" in self.model.lower() or "o1" in self.model.lower() or "o3" in self.model.lower():
                # GPT-5-nano only supports temperature=1.0, adjust if needed
                if temperature != 1.0:
                    temperature = 1.0
                
                # GPT-5-nano doesn't support stop parameter
                stop = None
                
                # For GPT-5 reasoning models, use reasoning_effort to control reasoning tokens
                kwargs = {
                    'model': self.model,
                    'messages': messages,
                    'temperature': temperature,
                    'max_completion_tokens': max_tokens,
                    'reasoning_effort': "low",
                    'stop': stop
                }
                if top_p is not None:
                    kwargs['top_p'] = top_p
                response = self.client.chat.completions.create(**kwargs)
            # GPT-4o and GPT-4 Turbo use max_completion_tokens but NOT reasoning_effort
            elif "gpt-4o" in self.model.lower() or "gpt-4-turbo" in self.model.lower():
                kwargs = {
                    'model': self.model,
                    'messages': messages,
                    'temperature': temperature,
                    'max_completion_tokens': max_tokens,
                    'stop': stop
                }
                if top_p is not None:
                    kwargs['top_p'] = top_p
                response = self.client.chat.completions.create(**kwargs)
            else:
                kwargs = {
                    'model': self.model,
                    'messages': messages,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'stop': stop
                }
                if top_p is not None:
                    kwargs['top_p'] = top_p
                response = self.client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content
            if content is None:
                print(f"WARNING: API returned None content. Response: {response}")
                print(f"Finish reason: {response.choices[0].finish_reason if response.choices else 'No choices'}")
                return ""
            return content
            
        except Exception as e:
            print(f"ERROR: OpenAI API call failed: {e}")
            raise
    
    def chat_completion_with_logprobs(
        self,
        messages: List[Dict[str, str]],
        possible_labels: List[str],
        temperature: float = 0.7,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate chat completion with log probabilities for classification.
        
        OpenAI-specific implementation that extracts probabilities for each possible label.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            possible_labels: List of possible label strings
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            
        Returns:
            Dict with 'prediction' (str) and 'probabilities' (Dict[str, float])
        """
        try:
            # GPT-5 reasoning models (o1, o3) use max_completion_tokens and reasoning_effort
            if "gpt-5" in self.model.lower() or "o1" in self.model.lower() or "o3" in self.model.lower():
                # GPT-5-nano only supports temperature=1.0
                if temperature != 1.0:
                    temperature = 1.0
                
                # GPT-5-nano doesn't support stop parameter
                stop = None
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    reasoning_effort="low",
                    logprobs=True,  # Request log probabilities
                    top_logprobs=20,  # Get top 20 token probabilities
                    stop=stop
                )
            # GPT-4o and GPT-4 Turbo use max_completion_tokens but NOT reasoning_effort
            elif "gpt-4o" in self.model.lower() or "gpt-4-turbo" in self.model.lower():
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    logprobs=True,  # Request log probabilities
                    top_logprobs=20,  # Get top 20 token probabilities
                    stop=stop
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=True,  # Request log probabilities
                    top_logprobs=20,  # Get top 20 token probabilities
                    stop=stop
                )
            
            # Extract prediction
            prediction = response.choices[0].message.content
            if prediction is None:
                prediction = ""
            
            # Extract log probabilities
            logprobs_data = response.choices[0].logprobs
            
            if logprobs_data and logprobs_data.content:
                # Get probabilities for the first token (the label prediction)
                first_token_logprobs = logprobs_data.content[0].top_logprobs
                
                # Build probability dictionary for possible labels
                label_probs = {}
                total_prob = 0.0
                
                # First, collect probabilities for labels that appear in top_logprobs
                for logprob_entry in first_token_logprobs:
                    token = logprob_entry.token.strip().lower()
                    logprob = logprob_entry.logprob
                    prob = 2.71828 ** logprob  # exp(logprob) to get probability
                    
                    # Check if this token matches any of our labels
                    for label in possible_labels:
                        if token == label.lower() or token in label.lower() or label.lower() in token:
                            if label not in label_probs or prob > label_probs[label]:
                                label_probs[label] = prob
                                total_prob += prob
                
                # Normalize probabilities
                if total_prob > 0:
                    for label in label_probs:
                        label_probs[label] = label_probs[label] / total_prob
                
                # For labels not in top logprobs, assign small probability
                remaining_prob = max(0, 1.0 - sum(label_probs.values()))
                missing_labels = [l for l in possible_labels if l not in label_probs]
                
                if missing_labels:
                    small_prob = remaining_prob / len(missing_labels)
                    for label in missing_labels:
                        label_probs[label] = small_prob
                
                # Normalize again to ensure sum = 1.0
                total = sum(label_probs.values())
                if total > 0:
                    label_probs = {k: v/total for k, v in label_probs.items()}
                
                # Prepare serializable raw response for logging
                raw_response = {
                    'model': response.model,
                    'choices': [{
                        'message': {
                            'role': response.choices[0].message.role,
                            'content': response.choices[0].message.content
                        },
                        'finish_reason': response.choices[0].finish_reason,
                        'logprobs': {
                            'content': [{
                                'token': lp.token,
                                'logprob': lp.logprob,
                                'top_logprobs': [{'token': tlp.token, 'logprob': tlp.logprob} 
                                                for tlp in lp.top_logprobs]
                            } for lp in logprobs_data.content[:5]]  # Only first 5 tokens for brevity
                        } if logprobs_data and logprobs_data.content else None
                    }],
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens if response.usage else None,
                        'completion_tokens': response.usage.completion_tokens if response.usage else None,
                        'total_tokens': response.usage.total_tokens if response.usage else None
                    }
                }
                
                return {
                    'prediction': prediction.strip(),
                    'probabilities': label_probs,
                    'raw_response': raw_response
                }
            
            # Fallback: no logprobs available
            return {
                'prediction': prediction.strip(),
                'probabilities': None,
                'raw_response': None
            }
            
        except Exception as e:
            print(f"WARNING: Failed to get logprobs: {e}")
            # Fallback to regular completion
            prediction = self.chat_completion(messages, temperature, max_tokens, stop)
            return {
                'prediction': prediction,
                'probabilities': None,
                'raw_response': None
            }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using OpenAI's tokenizer"""
        return len(self.encoding.encode(text))


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider (using new google-genai SDK)"""
    
    def __init__(self, config: dict):
        """
        Initialize Gemini provider.
        
        Args:
            config: Gemini configuration from config.yaml
        """
        try:
            from google import genai
            from google.genai import types
            
            api_key = config.get('api_key')
            if not api_key or api_key == 'YOUR_GEMINI_API_KEY_HERE':
                raise ValueError(
                    "Gemini API key not configured. Add your key to config.yaml"
                )
            
            self.client = genai.Client(api_key=api_key)
            self.model_name = config.get('model', 'gemini-2.5-flash')  # Use 2.5-flash with thinking disabled
            print(f"INFO: Initialized Gemini provider with model: {self.model_name}")
            print(f"INFO: Thinking mode disabled (thinking_budget=0)")
            
        except ImportError:
            raise ImportError(
                "Gemini package not installed. Install with: pip install google-genai"
            )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
        top_p: Optional[float] = None
    ) -> str:
        """Generate chat completion using Gemini"""
        try:
            from google.genai import types
            
            # Convert messages to Gemini format
            contents = []
            system_instruction = None
            
            for msg in messages:
                role = msg['role']
                content = msg['content']
                
                if role == 'system':
                    system_instruction = content
                elif role == 'user':
                    contents.append(types.Content(
                        role='user',
                        parts=[types.Part(text=content)]
                    ))
                elif role == 'assistant':
                    contents.append(types.Content(
                        role='model',
                        parts=[types.Part(text=content)]
                    ))
            
            # Build generation config with thinking DISABLED
            config_kwargs = {
                'temperature': temperature,
                'max_output_tokens': max_tokens,
                'system_instruction': system_instruction if system_instruction else None,
                'stop_sequences': stop if stop else None,
                'thinking_config': types.ThinkingConfig(thinking_budget=0)
            }
            if top_p is not None:
                config_kwargs['top_p'] = top_p
            
            config = types.GenerateContentConfig(**config_kwargs)
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            # Gemini response has a direct .text attribute
            if hasattr(response, 'text') and response.text:
                return response.text
            
            # If text is None, might have hit MAX_TOKENS - check finish reason
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
                if 'MAX_TOKENS' in str(finish_reason):
                    print(f"WARNING: Gemini hit MAX_TOKENS limit ({max_tokens}). Consider increasing max_tokens in config.yaml")
            
            print(f"ERROR: Gemini returned no text")
            return None
            
        except Exception as e:
            print(f"ERROR: Gemini API call failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for Gemini models.
        Uses simple word-based estimation (roughly 1.3 tokens per word).
        """
        words = len(text.split())
        return int(words * 1.3)


def get_llm_provider(config: dict) -> LLMProvider:
    """
    Factory function to create LLM provider based on config.
    
    Args:
        config: Full configuration dict from config.yaml
        
    Returns:
        Initialized LLM provider instance
    """
    provider_type = config['llm']['provider'].lower()
    
    if provider_type == 'ollama':
        return OllamaProvider(config['llm']['ollama'])
    elif provider_type == 'openai':
        return OpenAIProvider(config['llm']['openai'])
    elif provider_type == 'gemini':
        return GeminiProvider(config['llm']['gemini'])
    else:
        raise ValueError(
            f"Unknown provider: {provider_type}. Supported: 'ollama', 'openai', 'gemini'"
        )
