from ragas import SingleTurnSample,  EvaluationDataset, evaluate
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextPrecisionWithoutReference ,Faithfulness, AnswerRelevancy, LLMContextRecall, FactualCorrectness
from llm_models import LLModels
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import numpy as np
from ragas.run_config import RunConfig

class RAGAS:
    def __init__(self, rules, query, retrieval_context, response, groundtruth):
        self.rules = rules

        self.query = query
        self.retr_context = retrieval_context
        self.response = response
        self.groundtruth = groundtruth

        self.llm = LangchainLLMWrapper(LLModels().llm)
        self.embed = LangchainEmbeddingsWrapper(LLModels().embedding_model)
        self.combinations = ['AnswerRelevancy']

    def prepare_final_rules(self):
        final_rules = []
        for rl in self.rules:
            if rl in self.combinations:
                final_rules.append(eval(rl)(llm=self.llm,embeddings=self.embed))
            else:
                final_rules.append(eval(rl)(llm=self.llm))
        return final_rules
    
    def rag_eval(self,final_rules_metrics):
        sample = SingleTurnSample(
            user_input=self.query,
            response=self.response,
            retrieved_contexts=[self.retr_context],
            reference=self.groundtruth
        )

        dataset = EvaluationDataset(samples=[sample])
        result = evaluate(dataset=dataset, metrics=final_rules_metrics, run_config=RunConfig(timeout=1800))
        return result.to_pandas().replace(np.nan,0).to_dict()

    def call_funcs(self):
        print("RAGASm1", self.rules)
        final_rules_metrics = self.prepare_final_rules()
        report = self.rag_eval(final_rules_metrics)
        
        final_report = {}
        for ind,val in report.items():
            if ind == "retrieved_contexts":
                final_report[ind] = val[0][0]
            else:
                final_report[ind] = val[0]

        # report['llm_context_precision_without_reference'][0] = report['llm_context_precision_without_reference'][0]#.replace('NaN',"0.9487777")

        # report['faithfulness'][0] = report['faithfulness'][0]#.replace('NaN',"1")

        # report['answer_relevancy'][0] = report['answer_relevancy'][0]#.replace('NaN',"0.999999")
        return final_report