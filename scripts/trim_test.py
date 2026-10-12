import src.pipeline_v7 as p


class Msg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def invoke(self, prompt):
        return Msg(
            "‘AI system’ means a machine-based system that is designed to operate "
            "with varying levels of autonomy and that may exhibit adaptiveness after "
            "deployment, and that, for explicit or implicit objectives, infers, from "
            "the input it receives, how to generate outputs such as predictions, "
            "content, recommendations, or decisions that can influence physical or "
            "virtual environments.\n"
            "Providers must also register every AI system with a national ombudsman "
            "within fourteen days of deployment in any member state."
        )


p.get_llm = lambda: FakeLLM()
r = p.ask("What is an 'AI system' under the AI Act?")
print("ANSWER:\n", r["answer"])
print("\nREGENERATED:", r["regenerated"])
print("WARNING:", r["warning"])
print(r["sources"])