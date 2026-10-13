import src.pipeline_v7 as p


class Msg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def invoke(self, prompt):
        return Msg(
            "I don't know. The context does not contain information about this topic. "
            "The Digital Services Act (Regulation (EU) 2022/2065) is not covered by "
            "the provided excerpts."
        )


p.get_llm = lambda: FakeLLM()
r = p.ask("What capital requirements apply to banks under Basel III?")
print("ANSWER:\n", r["answer"])
print(r["sources"])
print("WARNING:", r["warning"])