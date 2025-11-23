from langchain_ollama import OllamaLLM


class LLM:
    def __init__(self):
        self.llm = OllamaLLM(model="qwen3:8b", temperature=0.5, seed=42)

    def generate_stream(self, prompt: str, stop_words: list[str] = []):
        response = self.llm.stream(prompt, stop=stop_words)
        return response

    def generate_judge_stream(self, prompt: str):
        return self.generate_stream(prompt, stop_words=["HUMAN:", "COMPUTER:"])

    def generate_computer_stream(self, prompt: str):
        return self.generate_stream(prompt, stop_words=["HUMAN:", "JUDGE:"])


llm = LLM()
