class EarningsExample:
    def __init__(self, example):
        self.example = example

    def get_prompt(self):
        return self.make_prompt()

    def get_query(self):
        prompt = self.get_company_info()
        prompt += self.example["question"]

        return prompt

    def make_prompt(self):
        prompt = "You are an expert analyst from Goldman Sachs with 15 years of experience."
        prompt += " Consider the following company: \n"
        prompt += "==========================\n"
        prompt += self.get_company_info()
        prompt += "==========================\n"
        prompt += "Answer the following question: \n"
        prompt += self.example["question"]
        return prompt


    def get_company_info(self):
        prompt = f"Date of the call: {self.example['date']}\n"
        prompt += f"Ticker: {self.example['ticker']}\n"
        prompt += f"Quarter: {self.example['q']}\n"

        return prompt