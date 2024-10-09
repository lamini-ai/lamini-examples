<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>

# LLM Classifier

Train a new classifier with just a prompt. No data needed -- but add data to boost, if you have it.

In this example, we build a classifier to remove low quality training data.

```python
import lamini

lamini.gate_pipeline_batch_completions = True

# Instatiate Classifier object to pass prompts to the LLM on the compute server
llm = lamini.LaminiClassifier()

prompts={
  "correct": "Questions with correct answers.",
  "incorrect": "Questions with incorrect answers."
}

llm.prompt_train(prompts)

llm.save("my_model.lamini")
```

Then, predict!
```
llm.predict(["Q: What is 2+2? A: The answer is 3."])
>> ["incorrect"]

llm.predict([
    "Q: What is 2+2? A: The answer is 3.",
    "Q: What is 2+2? A: The answer is 4.",
])
>> ["incorrect", "correct"]
```

### Optionally, add any data.

This can help with improving your classifier. For example, if the LLM is ever wrong:
```
llm.predict(["Q: What is 2+3? A: The answer is 5."])
>> ["incorrect"] # wrong!
```

#### You can correct the LLM by adding those examples as data.

And your LLM classifier will learn it:
```
llm = LaminiClassifier()

llm.add_data_to_class("correct", "Q: What is 2+3? A: The answer is 5.")

llm.prompt_train(prompts)

llm.predict(["Q: What is 2+3? A: The answer is 5."])
>> ["correct"] # correct!
```

If you include data on classes that aren't in your `classes`, then the classifier will include them as new classes, and learn to predict them. However, without a prompt, it won't have a description to use to further boost them.

General guideline: if you don't have any or little data on a class, then make sure to include a good prompt for it. Like prompt-engineering any LLM, creating good descriptions---e.g. with details and examples---helps the LLM get the right thing.

#### You can also work with this data more easily through files.
```
# Load data
llm.saved_examples_path = "path/to/examples.jsonl" # overrides default at /tmp/saved_examples.jsonl
llm.load_examples()

# Print data
print(llm.get_data())

# Train on that data (with prompts)
llm.prompt_train(prompts)

# Just train on that data (without any prompts)
llm.train()
```

Finally, you can save the data.
```
# Save data
llm.saved_examples_path = "path/to/examples.jsonl"
llm.save_examples()
```

Format of what `examples.jsonl` looks like:
```
{"class_name": "correct", "examples": ["Q: What is 2+3? A: The answer is 5."]}
{"class_name": "incorrect", "examples": ["Q: What is 2+3? A: The answer is 4."]}
```

# How LLM Classifiers Work

This section explains how the LLM Classifier works using a line-by-line implementation.

1. Data Generation - Many example data points are generated for each class using the LLM.
2. Embeddings - An LLM generates an embedding vector for each data point.
3. Training - A supervised learning classifier - e.g. logistic regression, BERT, etc - is trained on the embeddings.
4. Calibration - A precision-recall curve is used to set the thresholds of the classifier.

## 1. Data Generation

LLMs are called generative AI because they can easily generate data from a prompt.
We use this ability to generate training data for each class.  We include a three step
pipeline to generate training data.

1. Example Generation - Generate short descriptions of training examples, e.g. up to 10 words.
2. Example Randomization - Take a random sample from step 1, and ask an LLM to modify them to make them more diverse.
3. Example Expansion - Generate complete examples from short descriptions.

Data generation is very computationally intensive.  Lamini uses a large cluster of GPUs to
generate a large amount of training data.

Consider modifying this pipeline to suit your application, e.g. by adding data sources, and transforming data.

```python
def create_new_example_generator(self, prompt, original_examples):
    example_generator = self.generator_from_prompt(
        prompt,
        config=self.config,
        model_name=self.model_name,
        batch_size=self.batch_size // 5,
    )
    example_modifier = self.example_modifier(
        config=self.config,
        model_name=self.model_name,
        batch_size=self.batch_size // 5,
    )
    example_expander = self.example_expander(
        prompt, config=self.config, model_name=self.model_name
    )

    examples = original_examples.copy()

    index = len(examples)

    while True:
        # Phase 1: Generate example types from prompt
        compressed_example_features = example_generator.generate_examples(
            seed=index, examples=examples
        )

        # Phase 2: Modify the features to be more diverse
        different_example_features = example_modifier.modify_examples(
            compressed_example_features
        )

        different_example_features = chain(
            different_example_features, compressed_example_features
        )

        different_example_features_batches = self.batchify(
            different_example_features
        )

        # Phase 3: Expand examples from features
        for features_batches in different_example_features_batches:
            expanded_example_batch = example_expander.expand_example(
                features_batches
            )

            for expanded_example in expanded_example_batch:
                logger.debug(
                    f"Generated example number {index} out of {self.augmented_example_count}"
                )

                index += 1
                examples.append(expanded_example)
                yield expanded_example

                if index >= self.augmented_example_count:
                    return

```

### 1. Example Generation

Create a batch of examples using an LLM.  It is helpful to prompt the LLM:

1. with a description of the class
2. with any existing example data
3. to generate multiple examples at once
4. to use short descriptions, e.g. max 10 words

Consider the following example prompt.

```python
        system_prompt = "You are a domain expert who is able to generate many different examples given a description."

        prompt = ""

        # Randomly shuffle the examples
        random.seed(seed)
        random.shuffle(examples)

        # Include examples if they are available
        if len(examples) > 0:
            selected_example_count = min(self.max_history, len(examples))

            prompt += "Consider the following examples:\n"

            for i in range(selected_example_count):
                prompt += "----------------------------------------\n"
                prompt += f"{examples[i]}"
                prompt += "\n----------------------------------------\n"

        prompt += "Read the following description carefully:\n"
        prompt += "----------------------------------------\n"
        prompt += self.prompt
        prompt += "\n----------------------------------------\n"

        prompt += f"Generate {self.example_count} different example summaries following this description. Each example summary should be as specific as possible using at most 10 words.\n"
```

Note that we use the JSON mode of Lamini to make it easy to parse out the 5 different examples.

```python
    runner = LlamaV2Runner(config=self.config, model_name=self.model_name)

    results = runner(
        prompt=prompts,
        system_prompt=system_prompt,
        output_type={
            "example_1": "str",
            "example_2": "str",
            "example_3": "str",
            "example_4": "str",
            "example_5": "str",
        },
    )
```

### 2. Example Randomization

A good dataset for a classifier includes many different examples of the target class.  This is where
example randomization comes in.  It samples of group of generated examples, and asks the LLM to process
them again to make them more diverse.  We have found that this method is more effective than using a decoder
parameter such as temperature, which can increase randomness but also reduce LLM quality.

Consider the following example prompt.

```python
        system_prompt = "You are a domain expert who is able to clearly understand these descriptions and modify them to be more diverse."

        # Randomly shuffle the examples
        random.sample(examples)

        prompt = "Read the following descriptions carefully:\n"
        prompt += "----------------------------------------\n"
        for index, example in enumerate(examples[:example_count]):
            prompt += f"{index + 1}. {example}\n"
        prompt += "\n----------------------------------------\n"

        prompt += "Generate 5 more examples that are similar, but substantially different from those above. Each example should be as specific as possible using at most 10 words.\n"

```

### 3. Example Expansion

Now that we have many diverse examples, it is time to expand them into complete training examples.  See the prompt below to understand how to generate examples.

```python

    system_prompt = "You are a domain expert who is able to clearly understand this description and expand to a complete example from a short summary."

    prompt = "Read the following description carefully:\n"
    prompt += "----------------------------------------\n"
    prompt += self.prompt
    prompt += "\n----------------------------------------\n"
    prompt += "Now read the following summary of an example matching this description carefully:\n"
    prompt += "----------------------------------------\n"
    prompt += example
    prompt += "\n----------------------------------------\n"

    prompt += "Expand the summary to a complete example.  Be consistent with both the summary and the description.  Get straight to the point.\n"
```

## 2. Embeddings

Now that we have generated data for each class in the classifier, the next step is to start training
a classifier to draw a decision boundary between the classes.   We use another LLM to generate
a vector embedding for each item in the dataset.  This allows using the powerful reasoning
capabilities of the LLM for classification.

In Lamini, this is straightforward to do using the Embedding library.

```python
from lamini import Embedding

llm = Embedding()

examples = [...]

embeddings = llm.generate(examples)

```

Remember, embeddings are a 1-d vector representation of a chunk of text.  They are
a list of floats.

## 3. Training

Once we have vector embeddings for all of our training data.  The next step is to train
a classifier to map from embeddings into our classes.  There are many algorithms that could
be used here, and we recommend starting with [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).  LogisticRegression is nothing other than the last layer
of a neural network.  It is optimized for vector embedding inputs.  Optimizing just
one layer of a neural network is a much easier problem, so it almost always succeeds
and there is no need to fiddle with hyperparameters or look at training/eval curves.

```python
# Form the embeddings
X = []
y = []

for class_name, examples in tqdm(self.examples.items()):
    index = self.class_names_to_ids[class_name]
    y += [index] * len(examples)
    class_embeddings = self.get_embeddings(examples)
    X += class_embeddings

# Train the classifier
self.logistic_regression = LogisticRegression(random_state=0).fit(X, y)
```

Note: It is certainly possible to improve on LogisticRegression in this step.  More
advanced users may want to use DeepLearning, perhaps another transformer based
model, e.g. BERT, at this point.

## 4. Calibration

Finally, we have a classifier.  It takes a string as an input, and outputs a
score for each class.  What do you do with the score?

In this final step, we calibrate the model so that we can make decisions
using the scores from the model.

Let's say that we want to make sure that we only keep the answers that we are
90% confident are correct.  What would we set the score to?  Scores are normalized
between 0.0 and 1.0.  If we increase the threshold, that is like imposing a higher
bar on the samples, so we should end up with fewer examples that we are more
confident about.  How do we quantify that?

We can use a precision recall curve.  This curve plots precision vs recall at
different thresholds.

Precision is how many correct answers the model predicted.  Let's say we want
90% of the answers selected by our model to be correct, we should target a precision of
90%.

Recall is how many correct answers the model found.  Let's say that we want to
find 50% of all correct answers to include in our training dataset.  Then we
should set recall to 50%.

Precision and recall tradeoff against each other.  A great model will have
high precision and high recall, but a more modest model may have to sacrifice
high recall to achieve high precision.

You can see the tradeoff by plotting the precision/recall curve.

```python
from sklearn.metrics import PrecisionRecallDisplay

def plot_precision_recall_curve(examples):
    y_true = []
    y_score = []

    for example in examples:
        if not "predictions" in example:
            continue

        if not "label" in example:
            continue

        y_true.append(example["label"])
        y_score.append(get_positive_class(example["predictions"])["prob"])

    print("plotting precision-recall curve using {} examples".format(len(y_true)))

    display = PrecisionRecallDisplay.from_predictions(y_true, y_score)

    # Save the plot to a file.
    display.figure_.savefig("/app/lamini-classify/data/precision-recall-curve.png")
```

---

</div>
<div align="center">

![GitHub forks](https://img.shields.io/github/forks/lamini-ai/lamini-sdk) &ensp; © Lamini. &ensp; ![GitHub stars](https://img.shields.io/github/stars/lamini-ai/lamini-sdk)

</div>

--------
