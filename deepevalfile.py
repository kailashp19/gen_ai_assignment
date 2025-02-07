from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from llm_models import LLModels
#from langchain_openai import AzureChatOpenAI

class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"

class DeepEval:
    def __init__(self, rules, query, retrieval_context, response, groundtruth):
        self.rules = rules
        self.query = query
        self.retr_context = retrieval_context
        self.response = response
        self.groundtruth = groundtruth

        self.custom_model = LLModels().llm
        self.azure_openai = AzureOpenAI(model=self.custom_model)
        print("DeepEval")

    def prepare_final_rules(self):
        final_rules = []
        for rl in self.rules:
            # if rl in self.combinations:
            #     final_rules.append(eval(rl)(llm=self.llm,embeddings=self.embed))
            # else:
            final_rules.append(eval(rl)(model=self.azure_openai))
        return final_rules
    
    def call_funcs(self):
        final_rules_metrics = self.prepare_final_rules()
        test_cases = []
        #for index, row in df.iterrows():
        test_case = LLMTestCase(
            input=self.query,
            actual_output=self.response,
            expected_output=self.groundtruth,
            retrieval_context=[self.retr_context],
        )
        test_cases.append(test_case)

        # final_set = {}
        # for rind, rul in enumerate(self.rules):
        #     final_set[rul] = {}
        #     metricname = final_rules_metrics[rind]
        #     for sind, samp in enumerate(test_cases):
        #         metricname.measure(samp)
        #         final_set[rul][sind] = metricname.score

        final_set = {}
        for rind, rul in enumerate(self.rules):
            final_set[rul] = ""
            metricname = final_rules_metrics[rind]
            for samp in test_cases:
                metricname.measure(samp)
                final_set[rul] = metricname.score
        
        final_set['UserInput'] = self.query
        final_set['Retrieval_context'] = self.retr_context
        final_set['Response'] = self.response
        final_set['GroundTruth'] = self.groundtruth

        return final_set