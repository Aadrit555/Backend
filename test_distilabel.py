import os
from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration
from backend.config import settings

def main():
    print("Testing distilabel 1 example...")
    os.environ["OPENAI_API_KEY"] = settings.openrouter_api_key
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    
    with Pipeline(name="Test") as pipeline:
        loader = LoadDataFromDicts(
            name="load",
            data=[{"instruction": f"Test 0"}]
        )
        generator = TextGeneration(
            name="gen",
            llm=OpenAILLM(model="nvidia/nemotron-3.5-lightning:free"),
        )
        loader >> generator
    
    print("Running...")
    distiset = pipeline.run(use_cache=False)
    print("Done running pipeline!")
    print("Type of default:", type(distiset["default"]))
    print("First item:", distiset["default"][0])

if __name__ == "__main__":
    main()
