# TakeMeter project plan

I want to fine-tune a model to predict the verdict an AITA post would receive from Reddit. This is a class project using 200 labeled posts, the course Colab notebook, and a Groq baseline.

## Community

I chose r/AmItheAsshole because people post personal conflicts and ask other readers who was in the wrong. The topics vary, but the four main verdicts give me a clear classification task. I am interested in whether a model can predict the community's response from the post alone.

The target is the community verdict, not my own opinion. That also means the labels can reflect disagreements and biases in the community.

## Labels and examples

| Label | Meaning | Two examples |
| --- | --- | --- |
| **YTA — You're the A-Hole** | The author is in the wrong and the other person is not. | A teenager refuses to clean his room and tells his mother to leave him alone. A new driver secretly buys a car after being confronted about driving dangerously. |
| **NTA — Not the A-Hole** | The author is not in the wrong, but the other person is. | A tenant changes the Wi-Fi password after a landlord repeatedly enters without notice. An employee refuses to charge for her own candy after coworkers pressure her to fundraise. |
| **ESH — Everyone Sucks Here** | Both the author and the other person have done something wrong. | A couple argues over sharing a burger, followed by the silent treatment. A sibling argument about an autism diagnosis turns into a hurtful confrontation on both sides. |
| **NAH — No A-Holes Here** | Neither person is in the wrong, even if they disagree or feel hurt. | Someone wants to travel alone while their fiancé feels left out. Someone considers missing a friend's birthday weekend but still plans to celebrate on the actual day. |

These labels separate four outcomes: the author is wrong, the other person is wrong, both are wrong, or neither is wrong. I will assign one label per post. They should cover at least 90% of ordinary judgment posts, although I have not measured that coverage. INFO posts, unresolved threads, removed posts, and updates will be excluded rather than put in an “other” category.

## Hard edge cases

The hardest case I expect is **NTA versus NAH**: the author sets a reasonable boundary, but the other person is hurt. Feeling disappointed is not enough by itself to make someone wrong. Insults, pressure, retaliation, or ignoring an obligation may change that.

For example, refusing a friend's offer of a travel loan could be NAH if the friend only feels rejected. The collected post received NTA, so I kept that label and noted the ambiguity. When my interpretation differs from the community, I will use the resolved verdict rather than replace it with my own.

Other boundaries to watch:

- **YTA versus ESH:** A coworker reacts sharply after being asked to record her name without an accent. I need to distinguish an understandable reaction from a separate wrong. The recorded label is YTA.
- **NTA versus ESH:** A friend talks through movies, and the group secretly excludes her from plans. Bluntly explaining the problem is one thing; how the group handled it is another. The recorded label is ESH.
- **NAH versus blame:** Two friends have different expectations about time and emotional support. Hurt feelings alone do not settle who is wrong. A post about repaying a former friend without answering her emotional message is an example of this uncertainty.

For “WIBTA” posts, I will judge the proposed action using the same definitions. A hypothetical situation is not automatically NAH. If the thread has no clear verdict, I will leave it out and collect another example.

## Data collection plan

I will manually collect public AITA posts from Reddit, keeping the title and body as the input. I will read each post and use its verdict flair, or the top verdict comment if no flair is available, for the label. I will not include comments, scores, usernames, or flair in the input text.

My target is **200 posts: 50 NTA, 50 YTA, 50 ESH, and 50 NAH**. If a label is short after collecting 200, I will look for more posts with that verdict until each label has enough examples. I will check for empty text and duplicates, and keep notes on difficult cases for the README.

The full dataset will be saved as one CSV with `text` and `label` columns. The notebook will create shuffled, stratified splits: 70% training, 15% validation, and 15% test, using seed 42. With 200 posts, that gives 140 training, 30 validation, and 30 test examples.

## Training and comparison

I will start with `distilbert-base-uncased` in Colab and use the validation results to choose training settings. I will keep the test set separate for the final evaluation. Posts longer than 256 tokens will be cut off, which could remove details that affect the verdict.

The assignment's baseline is a zero-shot prompt to Groq's `meta-llama/llama-4-scout-17b-16e-instruct`. Both models should be evaluated on the same test posts.


## Evaluation metrics and success

I will report:

- **Accuracy:** How many test verdicts each model gets right overall.
- **Per-class precision, recall, and F1:** Whether the model handles all four verdicts or mainly succeeds on one or two.
- **Macro-F1:** The average F1 across the four labels, giving each equal weight.
- **Confusion matrix:** Which labels get mixed up, especially NTA/NAH and YTA/ESH.

Accuracy alone will not tell me whether the model has trouble recognizing shared fault or situations where nobody is wrong. The per-class results and matrix should help explain those mistakes. There are only 7–8 test examples per label, so I will be careful about generalizing from a few predictions.

My basic goal is **at least 35% accuracy, macro-F1 of at least 0.35, and F1 above zero for every label**. My stronger target is **at least 47% accuracy, macro-F1 of at least 0.45, and no label below 0.25 F1**.

For a useful draft-feedback tool, I would also want at least **70% accuracy on whether the author is at fault** (YTA/ESH versus NTA/NAH). That would still need testing on more posts before I considered it ready for regular use.

These goals can be checked against the saved predictions. I will report a missed target rather than lower it to match the result. Beating the Groq baseline would be encouraging, but I will report the comparison even if fine-tuning does worse.

## AI Tool Plan

**Label stress-testing:** Ask Claude for 5–10 made-up posts that fall between two labels, then try the definitions on them. If a case exposes an unclear distinction, revise the wording. These examples are only for checking the definitions, not for training. The earlier draft records eight examples and changes covering hurt feelings, public reactions, and hypothetical actions.

**Annotation assistance:** Do not use an LLM to pre-label the data. Read the posts and record the community verdict manually, since an AI's own judgment could differ from the target label. Disclose this choice in the README.

**Failure analysis:** Give an AI tool the wrong predictions and ask which mistakes repeat. Check its suggestions against the confusion matrix and full post text. Keep patterns supported by the counts, and describe ideas about causes as possibilities unless I have tested them. Record any claims I reject or revise in the README.

## Stretch feature: error pattern analysis

I will group mistakes by true and predicted label and look for repeated confusions. I will read examples from the largest groups and consider whether the issue is an unclear label, missing context, or too few examples of that distinction.

The README will include the counts, three specific wrong predictions, possible explanations, and what I would change next.
