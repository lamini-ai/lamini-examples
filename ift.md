# Instruction Fine Tuning

## Introduction

Instruction Fine-Tuning is used to make language models better at following instructions by training them on examples where instructions are paired with desired outcomes.

Consider this prompt:
```
Generate a recipe suggestion for a user interested in trying a vegetarian dish.
```

Without fine tuning, the model may respond with a recipe
without considering the user's preferences or dietary restrictions. :thumbsdown:

Let's try fine tuning with some example pairs.

```
[{"instruction": "Provide a vegetarian recipe option for someone who enjoys spicy flavors, with a focus on quick preparation and minimal ingredients.",
 "desired_output": "Encourage the model to consider user preferences for spice level, preparation time, and simplicity."
 },
 {"instruction": "Suggest a vegetarian recipe suitable for a user following a gluten-free diet, emphasizing diverse textures and flavors without compromising dietary restrictions."
 "desired_output": "Guide the model to consider specific dietary needs and preferences."
 }
]
```

Improved Response (after fine-tuning): The model, having learned from these fine-tuned examples, generates vegetarian recipe suggestions that align with users' preferences, ensuring a more personalized and enjoyable cooking experience.

TODO: add more, mayb example