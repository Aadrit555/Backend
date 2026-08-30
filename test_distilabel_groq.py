import os
import json
import re
from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration
from backend.config import settings
from dotenv import load_dotenv

load_dotenv("backend/.env")

def main():
    print("Testing distilabel Groq...")
    
    with Pipeline(name="Test") as pipeline:
        loader = LoadDataFromDicts(
            name="load",
            data=[{"instruction": "Extract a conversational Q&A pair from this text. Return ONLY a JSON dictionary with 'q' and 'a'.\nTEXT: This is a test."}]
        )
        generator = TextGeneration(
            name="gen",
            llm=OpenAILLM(
                model="openai/gpt-oss-120b",
                base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY_1", "")
            ),
        )
        loader >> generator
    
    print("Running...")
    distiset = pipeline.run(use_cache=False)
    print("Done running pipeline!")
    
    qa_pairs = []
    if "default" in distiset and "train" in distiset["default"]:
        for row in distiset["default"]["train"]:
            output_text = row.get("generation", "")
            print("Output text:", output_text)
            try:
                match = re.search(r'\{.*\}', output_text, re.DOTALL)
                if match:
                    pair = json.loads(match.group(0))
                    if "q" in pair and "a" in pair:
                        qa_pairs.append(pair)
            except Exception as e:
                print("JSON parsing error:", e)
    
    print("Generated pairs:", len(qa_pairs))

if __name__ == "__main__":
    main()
