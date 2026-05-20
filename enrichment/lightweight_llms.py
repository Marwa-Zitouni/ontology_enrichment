
import time
import torch
from typing import Optional, Dict, Any
import json
import re


# Lightweight Model Configurations
LIGHTWEIGHT_MODELS = {
    "Qwen2.5-1.5B-Instruct": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "params": 1.5e9,
        "context_length": 32768,
        "recommended_device": "auto",
        "max_new_tokens": 1024,
        "quantization": "int4",  # 4-bit quantization for efficiency
        "temperature": 0.7,
        "description": "Fastest lightweight model, suitable for edge deployment"
    },
    "Mistral-7B-Instruct": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "params": 7e9,
        "context_length": 8192,
        "recommended_device": "auto",
        "max_new_tokens": 1024,
        "quantization": "int8",  # 8-bit for balance
        "temperature": 0.7,
        "description": "Balanced model with strong reasoning capabilities"
    },
    "Phi-3-Mini": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "params": 3.8e9,
        "context_length": 4096,
        "recommended_device": "auto",
        "max_new_tokens": 1024,
        "quantization": "int4",
        "temperature": 0.7,
        "description": "Efficient model optimized for inference speed"
    },
}


# Enhanced LLM Enricher with Lightweight Support

class LightweightLLMEnricher:
    

    def __init__(
        self,
        model_name: str = "Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        quantize: bool = True,
        use_mock: bool = False,
    ):
       
        self.model_name = model_name
        self.use_mock = use_mock
        self.quantize = quantize
        self.device = device

        # Get config for this model
        if model_name not in LIGHTWEIGHT_MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        self.config = LIGHTWEIGHT_MODELS[model_name]
        self.model_id = self.config["model_id"]

        # Initialize tracking
        self.pipeline = None
        self.metrics = {
            "load_time": None,
            "last_inference_time": None,
            "total_inference_time": 0.0,
            "num_inferences": 0,
        }

        if not use_mock:
            self._load_model()

    def _load_model(self):
        import psutil
        proc = psutil.Process()
        mem_before = proc.memory_info().rss / 1024 / 1024

        try:
            from transformers import pipeline as hf_pipeline, BitsAndBytesConfig
        except ImportError:
            print("transformers library not found. Installing...")
            import subprocess
            subprocess.check_call([
                "pip", "install", "transformers", "accelerate", "bitsandbytes"
            ])
            from transformers import pipeline as hf_pipeline, BitsAndBytesConfig

        start_time = time.time()

        print(f"\n[LOADING] {self.model_name} ({self.config['params']/1e9:.1f}B params)")
        print(f"  Model ID: {self.model_id}")
        print(f"  Context: {self.config['context_length']} tokens")
        print(f"  Max output: {self.config['max_new_tokens']} tokens")

        try:
            # Determine device
            device_map = self.device
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # Apply quantization if enabled
            quantization_config = None
            if self.quantize and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    if self.config["quantization"] == "int4":
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                    elif self.config["quantization"] == "int8":
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                        )
                except ImportError:
                    print("  [NOTE] BitsAndBytes not available, using standard precision")

            # Load pipeline
            self.pipeline = hf_pipeline(
                "text-generation",
                model=self.model_id,
                device_map=device_map,
                dtype=torch_dtype,
                quantization_config=quantization_config,
                model_kwargs={"trust_remote_code": True},
                max_new_tokens=self.config["max_new_tokens"],
                do_sample=True,
                temperature=self.config["temperature"],
            )

            load_time = time.time() - start_time
            self.metrics["load_time"] = load_time

            mem_after = proc.memory_info().rss / 1024 / 1024
            mem_used = mem_after - mem_before

            print(f"  [LOADED] {load_time:.2f}s ({mem_used:.0f} MB)")

        except Exception as e:
            print(f"  [ERROR] Failed to load model: {e}")
            print(f"  [FALLBACK] Using mock mode")
            self.use_mock = True

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text from the LLM.

        Args:
            prompt: Input text
            max_tokens: Override max tokens (default from config)

        Returns:
            Generated text
        """
        if self.use_mock:
            return self._mock_generate(prompt)

        start_time = time.time()

        try:
            # Use chat format for instruction-tuned models
            messages = [{"role": "user", "content": prompt}]

            output = self.pipeline(
                messages,
                return_full_text=False,
                max_new_tokens=max_tokens or self.config["max_new_tokens"],
            )

            # Extract text
            if isinstance(output, list) and len(output) > 0:
                if "generated_text" in output[0]:
                    result = output[0]["generated_text"]
                else:
                    result = str(output[0])
            else:
                result = str(output)

            # Track metrics
            inference_time = time.time() - start_time
            self.metrics["last_inference_time"] = inference_time
            self.metrics["total_inference_time"] += inference_time
            self.metrics["num_inferences"] += 1

            return result

        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return ""

    def _mock_generate(self, prompt: str) -> str:
        """Mock generation for testing."""
        ns = "http://example.org/battery-ontology#"
        return json.dumps({
            "new_anomaly_types": [
                {
                    "name": "RapidDegradation",
                    "label": "Rapid Degradation",
                    "description": "Accelerated capacity loss",
                    "parent_class": "FailureMode",
                }
            ],
            "new_causal_relationships": [
                {
                    "from": f"{ns}Overheating",
                    "to": f"{ns}RapidDegradation",
                    "description": "Heat accelerates degradation",
                }
            ],
            "new_properties": [],
            "rdf_triples": [],
        })

    def get_metrics(self) -> Dict[str, Any]:
        """Return collected metrics."""
        avg_inference = (
            self.metrics["total_inference_time"] / self.metrics["num_inferences"]
            if self.metrics["num_inferences"] > 0
            else 0
        )
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "params_billions": self.config["params"] / 1e9,
            "load_time_seconds": self.metrics["load_time"],
            "total_inference_time_seconds": self.metrics["total_inference_time"],
            "num_inferences": self.metrics["num_inferences"],
            "average_inference_time": avg_inference,
        }


# Compatibility Wrapper

class LLMEnricherAdapter:
    """
    Adapter that makes LightweightLLMEnricher compatible with
    the existing LLMOntologyEnricher interface.
    """

    def __init__(self, model_name: str, use_mock: bool = False):
        """
        Args:
            model_name: HuggingFace model ID or lightweight model key
            use_mock: Use mock responses
        """
        # Check if it's a lightweight model
        if model_name in LIGHTWEIGHT_MODELS:
            self.enricher = LightweightLLMEnricher(
                model_name=model_name,
                use_mock=use_mock
            )
        else:
            # Fall back to original enricher
            from enrichment.llm_enrichment import LLMOntologyEnricher
            self.enricher = LLMOntologyEnricher(
                model_name=model_name,
                use_mock=use_mock
            )

    def generate_enrichment(self, anomaly_info: Dict, semantic_context: Dict) -> Dict:
       
        # Build prompt (same format as original)
        from enrichment.llm_enrichment import build_enrichment_prompt
        prompt = build_enrichment_prompt(anomaly_info, semantic_context)

        # Generate response
        response_text = self.enricher.generate(prompt)

        # Parse JSON
        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> Dict:
        """Extract JSON from response."""
        try:
            # Try direct JSON parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try finding JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback
        return {
            "new_anomaly_types": [],
            "new_causal_relationships": [],
            "new_properties": [],
            "rdf_triples": [],
            "raw_response": response_text,
        }
