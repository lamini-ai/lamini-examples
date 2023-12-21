# Content Fine Tuning (DRAFT, INCOMPLETE)

Content fine-tuning is similiar to refining instructions.
However, the model is trained using an extensive set of question and answer pairs, often hundreds or thousands of pairs.
The objective is to input content and information into the model without causing response quality degradation.

There are three ways to obtain the question and answer pairs:
1. Use the same question and answer generation technique from the [instruction fine tuning guide](https://github.com/lamini-ai/sdk/blob/main/ift.md).
2. Use existing data if available. For example, data from actual customer and service agent chats.
3. Manually write question and answer pairs. 🚫 Please don't do this!

In this tutorial, we will describe how to best generate question and answer pairs for content fine tuning.
The key idea is to iterate and refine question-answer generation until you have high quality data.
You can start with question-answer generation on noisy data.

1. Generate a large set of question-answer pairs.
Select random pairs manually to assess their quality. You have the flexibility to set the quality threshold for these pairs. Achieving high quality might mean finding 1 bad pair out of every 10 pairs.

2. Generate a small set of question-answer pairs. If they are good enough, then ask the model to
generate more pairs.

3. Filter out bad question-answer pairs.
You can use prompt engineering to instruct the model to perform this task on your behalf.

4. Modify the chunks.

Chunk content usually affects the result more than chunk size.
Experiment with modifying chunk content.  Consider the fictional bill from congress:

```
Title: The "Banana Peel Recycling Act"

Preamble:
In recognition of the slippery situation our nation faces and the urgent need for sustainable solutions, this bill proposes a groundbreaking initiative to establish a national Banana Peel Recycling Program.

Section 1: Banana Peel Collection Bins
a. Mandate the installation of banana peel collection bins in public spaces, including parks, government offices, and, of course, comedy clubs.
b. Encourage citizens to toss banana peels into designated bins to promote a cleaner and more eco-friendly environment.

...

Section 7: Banana Peel Art Contest for Kids
...
y. Establish a panel of comedians and artists to select the most imaginative and hilarious banana peel artworks.
...

Section 8: Banana Peel Art Contest for Adults
...
z. Establish a panel of monkeys to select the most imaginative and hilarious banana peel artworks.
```

Suppose the text in each section is very long, so that the chunks containing parts of the text below do not contain the section number.
```
Establish a panel of monkeys to select the most imaginative and hilarious banana peel artworks.
```

The model may generate wrong answers for the queries below.
```
Which section establishes a panel of monkeys to select banana peel artworks?
Which contest establishes a panel of monkeys to selectbanana peel artworks?
```

To resolve this problem, you can simply update the chunking code to prepend the section number and
title to each chunk.