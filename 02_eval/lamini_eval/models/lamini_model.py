from lamini_prompt.lamini_model_stage import LaminiModelStage


def load_lamini_model(model_name):
    return LaminiModel(model_name)


class LaminiModel:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name is None:
            self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    def get_stages(self, dataset):
        return [LaminiModelStage(dataset=dataset, model_name=self.model_name)]
