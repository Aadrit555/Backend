import os
import json
from typing import Dict, Any, List, Optional
from groq import Groq

class LLMProvider:
    def generate_schema(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError
        
    def fill_semantic_fields(self, schema: Dict[str, Any], partial_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError

from backend.config import settings

class GroqProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.model = model or settings.groq_model
        
        self.keys = []
        if api_key:
            self.keys.append(api_key)
        else:
            for i in range(1, 10):
                k = os.environ.get(f"GROQ_API_KEY_{i}") or getattr(settings, f"groq_api_key_{i}", None)
                if k and k not in self.keys:
                    self.keys.append(k)
            k_std = os.environ.get("GROQ_API_KEY")
            if k_std and k_std not in self.keys:
                self.keys.append(k_std)
                
        if not self.keys:
            raise ValueError("No GROQ API keys found. Please set GROQ_API_KEY_1, etc. in .env.")
            
        self.clients = [Groq(api_key=k) for k in self.keys]
        self.client_idx = 0

    @property
    def client(self):
        return self.clients[self.client_idx]

    def rotate_client(self):
        self.client_idx = (self.client_idx + 1) % len(self.clients)
        print(f"Rotated to Groq API Key #{self.client_idx + 1}")

    def generate_schema(self, prompt: str) -> Dict[str, Any]:
        system_prompt = """You are an expert dataset architect. The user will describe a dataset they want to generate.
Your job is to output a JSON object describing the schema of this dataset.
Output EXACTLY this JSON structure and nothing else:
{
  "dataset_description": "A short summary of what this dataset is",
  "columns": [
    {
      "name": "column_name",
      "type": "string|integer|float|boolean|category|date|uuid",
      "description": "What this column represents",
      "generation_strategy": "python|llm",
      "constraints": {}
    }
  ],
  "python_code": "def generate_base_data(num_rows):\n    import random\n    import uuid\n    rows = []\n    for _ in range(num_rows):\n        # Strictly enforce tight, logical mathematical relationships here:\n        hours = random.uniform(0, 15)\n        marks = max(0, min(100, hours * 6.5 + random.randint(-5, 5)))\n        rows.append({'hours': hours, 'marks': marks})\n    return rows"
}
IMPORTANT:
- Use "python" generation_strategy for ALMOST ALL FIELDS (numerical, categorical, identifiers). You must write the logic inside the python_code string to generate these fields mathematically and deterministically.
- Use "llm" strategy ONLY for deep semantic text (e.g. realistic product reviews, complex sentences) that cannot be hardcoded or realistically generated via simple python rules.
- The `python_code` must define a function `def generate_base_data(num_rows):` that returns a list of dictionaries. The keys in the dictionaries must match the column names using the "python" strategy.
- Do NOT invoke LLM for simple conditional text (e.g., if marks > 80 then status = "Pass" should be done in python).
- KEEP VALUES GROUNDED AND REALISTIC: Who studies 12 hours a day? Keep the random ranges and mathematical coefficients grounded in real-world feasibility. Do not generate absurdly high or low numerical ranges unless requested. Rely heavily on FACTUAL REAL-WORLD AVERAGES (e.g. average baby birth weight is ~3.5kg, average adult height is ~170cm). Use your vast knowledge base to set highly accurate means and medians for your formulas.
- AVOID HARD BOUNDARIES (NO CLIPPING): Do NOT use `min()` or `max()` to constrain natural variables like length, weight, or age (e.g. never use `min(45, length)`). This ruins the data by creating artificial ceilings where 50% of your rows are exactly the same number! Instead, rely PURELY on `random.gauss(mean, std_dev)` to naturally constrain values within realistic bell curves. ONLY use `max(0, val)` to prevent impossible negative numbers, or `min(100, val)` for strict percentages.
- USE GAUSSIAN DISTRIBUTIONS: For continuous real-world variables (like height, weight, test scores), strongly prefer `random.gauss(mu, sigma)` instead of uniform distributions to create realistic bell curves.
- REALISTIC CATEGORICAL DISTRIBUTIONS: When generating categorical data (like Gender, Country, Job Title), DO NOT use uniform `random.choice`. You MUST use `random.choices(..., weights=[...])` with strictly realistic real-world statistical probabilities (e.g., Male 49%, Female 49%, Other 2%).
- DELICATE RANDOM NOISE: You are in charge of injecting random noise in the python code (e.g. `+ random.gauss(0, 2)`). Make sure the noise is proportionate to the values so the correlation isn't completely destroyed but remains highly realistic!
- Keep column names snake_case."""
        
        for attempt in range(len(self.clients)):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                err_msg = str(e).lower()
                if "rate_limit" in err_msg or "429" in err_msg or "413" in err_msg:
                    print(f"Rate limit on key #{self.client_idx+1}, rotating...")
                    self.rotate_client()
                else:
                    raise e
        raise Exception("All Groq API keys exhausted due to rate limits.")

    def fill_semantic_fields(self, schema: Dict[str, Any], partial_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        llm_cols = [c for c in schema.get("columns", []) if c.get("generation_strategy") == "llm"]
        if not llm_cols:
            return partial_records
            
        system_prompt = f"""You are a synthetic data generation engine. 
You will be given a list of JSON records that are partially filled with statistical data.
Your job is to populate the remaining empty fields based on the dataset description and schema.
The fields you need to generate are: {[c['name'] for c in llm_cols]}

CRITICAL: Ensure extremely high variance, creativity, and diverse sentence structures. Do NOT repeat the same phrasing, themes, or structures across records. Write as if completely different humans wrote each one.
Ensure your generated text makes logical sense with the rest of the provided fields in each record.
Output a JSON object with a single key "records" containing the fully populated list of records in the exact same order.
Make sure all text fields are varied and realistic."""

        for attempt in range(len(self.clients)):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps({"records": partial_records}, default=str)}
                    ],
                    model=self.model,
                    response_format={"type": "json_object"},
                    temperature=0.85,
                )
                result = json.loads(response.choices[0].message.content)
                return result.get("records", partial_records)
            except Exception as e:
                err_msg = str(e).lower()
                if "rate_limit" in err_msg or "429" in err_msg or "413" in err_msg:
                    print(f"Rate limit on key #{self.client_idx+1}, rotating...")
                    self.rotate_client()
                else:
                    raise e
        raise Exception("All Groq API keys exhausted due to rate limits.")
