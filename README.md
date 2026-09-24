# TakeMeter

For this project, I fine-tuned a model to predict how Reddit's AITA community would judge a post: NTA, YTA, ESH, or NAH. I wanted to see whether training on 200 labeled posts would help it recognize the differences between these verdicts.

The fine-tuned model got **7 out of 30 test posts right (23.3%)**. The Groq baseline got **14 out of 30 right (46.7%)**. Fine-tuning did not improve the results in this run.

**Demo video:** Still needs to be recorded and linked here (3–5 minutes).

## Community and labels

I chose r/AmItheAsshole because judging interpersonal conflicts is the main purpose of the community. The posts cover family, friendships, work, and other everyday situations, and the verdicts already have meanings that readers recognize. The goal is to predict the community's response, not decide who is objectively right.

| Label | Meaning | Two examples |
| --- | --- | --- |
| **YTA — You're the A-Hole** | The author is in the wrong and the other person is not. | A teenager refuses to clean his room and tells his mother to leave him alone. A new driver secretly buys a car after being confronted about driving dangerously. |
| **NTA — Not the A-Hole** | The author is not in the wrong, but the other person is. | A tenant changes the Wi-Fi password after a landlord repeatedly enters without notice. An employee refuses to charge for her own candy after coworkers pressure her to fundraise. |
| **ESH — Everyone Sucks Here** | Both the author and the other person have done something wrong. | A couple argues over sharing a burger, followed by the silent treatment. A sibling argument about an autism diagnosis turns into a hurtful confrontation on both sides. |
| **NAH — No A-Holes Here** | Neither person is in the wrong, even if they disagree or feel hurt. | Someone wants to travel alone while their fiancé feels left out. Someone considers missing a friend's birthday weekend but still plans to celebrate on the actual day. |

Each post gets one label. The main distinction is whether the author, the other person, both, or neither is at fault. These four labels should cover most judgment posts; I excluded INFO, unresolved, deleted, and update/meta posts rather than adding an “other” label. I did not measure coverage across the whole subreddit.

The hardest differences are NTA versus NAH, and YTA versus ESH. Being upset does not automatically make someone wrong. I also considered whether a response was understandable or went far enough to make both people responsible. For “WIBTA” posts, I considered the proposed action rather than assuming nobody was wrong because it had not happened yet.

## Dataset and labeling

The complete dataset is [posts.csv](posts.csv), with `text` and `label` columns. I manually copied public AITA posts, including their titles and bodies, from Reddit. Most were collected on September 16, 2026. Some use the sister community's “AITAH” wording, but they all come from the same AITA subreddit.

I read each post and recorded the community verdict from the flair, or the top verdict comment when there was no flair. I did not use an LLM to pre-label the posts. Comments, usernames, scores, and flair were not included as model inputs. When my own reading differed from the community's verdict, I kept the community label (generally the most popular response) because that is what I was trying to predict.

| Label | Posts | Share |
| --- | --- | --- |
| NTA | 50 | 25% |
| YTA | 50 | 25% |
| ESH | 50 | 25% |
| NAH | 50 | 25% |
| **Total** | **200** | **100%** |

I collected equal numbers so each label had enough examples. This is a balanced sample, not a measurement of how often each verdict appears on Reddit.

Three posts were especially difficult to label:

- **Refusing a friend's offer to lend $2,000 for a trip — NTA.** The friend felt hurt by the refusal, which could also fit NAH. I kept NTA to match the community verdict, although the difference is not clear from the label definitions alone.
- **Telling a friend why she is no longer invited to movies — ESH.** The friend kept talking during movies, but the group also hid plans from her in a separate chat. I kept ESH because both the disruption and the way the group handled it mattered.
- **Asking a coworker to record her name in Vocera without an accent — YTA.** There was a practical problem with the system recognizing the name, but the request was also personal and upsetting. I kept YTA rather than ESH because I read the coworker's sharp reply as a reaction to the repeated request.

## Fine-tuning

I used `distilbert-base-uncased` with a four-label classification head in the course Colab notebook, using a T4 GPU.

| Setting | Value |
| --- | --- |
| Training / validation / test | 140 / 30 / 30 posts |
| Split | Shuffled and stratified, with `random_state=42` |
| Maximum input length | 512 tokens; longer posts are cut off |
| Epochs | 8 |
| Learning rate | 3e-5 |
| Training batch size | 18 |
| Weight decay | 0.01 |
| Warmup steps | 50 |
| Model selection | Best validation accuracy, checked after each epoch |

The main setting I changed was the number of epochs. At 3 epochs, the model showed little progress. It seemed it was guessing NTA for almost every post, and getting around 24% of all answers correct. After AI suggested more training, I tried 17, but training loss fell close to zero while validation loss rose above 2.0. That suggested overfitting, so I reduced it to 8 and used a learning rate of 3e-5. The final run still performed poorly; this change did not solve the problem.

I used the 512-token limit to keep as much of each post as possible. Longer posts can still lose details near the end, which may matter when deciding who was at fault.

## Baseline

The assignment specified Groq's `meta-llama/llama-4-scout-17b-16e-instruct`. My project notes record that it was unavailable when I ran the baseline, so I used `openai/gpt-oss-120b` instead.

I sent each of the same 30 test posts to Groq with the prompt below. The notebook used temperature `0`, `max_tokens=1024`, and `reasoning_effort="low"`. It matched the responses to the four labels and compared them with the dataset labels. All 30 responses were parseable in the final run.

<details>
<summary>Full prompt used in the notebook</summary>

```text
You are a classifier for r/AmItheAsshole (and its sibling r/AITAH), a Reddit community where people describe an interpersonal conflict they were involved in and ask the community to judge who was in the wrong.

Your task: read one post and predict THE VERDICT THE COMMUNITY WOULD REACH — not your own moral opinion. You are modelling how a crowd of Reddit commenters would vote, including their tendencies: they side with clear, sympathetic narrators, they forgive blunt honesty when the substance is right, and they punish authors whose own account reveals them as unreasonable.

Every post is judged on two independent questions:
  1. Is the AUTHOR in the wrong?
  2. Is the OTHER PARTY in the wrong?

The four labels are the four answers:

yta — "You're the A-Hole." The author is in the wrong and the other person or people are not. If the situation hasn't happened yet, the action the author plans to take would make them in the wrong.

nta — "Not the A-Hole." The author is not in the wrong and the other person or people are.

esh — "Everyone Sucks Here." Both the author and the other party are in the wrong, and each did something wrong that stands on its own rather than being a reaction to the other.

nah — "No A-Holes Here." Nobody described is in the wrong. The conflict comes from a genuine clash of needs, expectations, or information, where every party behaved reasonably given what they knew.

WHAT COUNTS AS BEING IN THE WRONG

A party is in the wrong only if they DID one of these:
- demanded, forbade, or issued an ultimatum (not merely asked or objected)
- insulted or name-called — attacked the person, not the position
- retaliated, imposing a consequence beyond simply declining to participate
- deceived, or concealed something that disadvantaged the other party
- withdrew normal relations as punishment (silent treatment, stonewalling, punitive exclusion)
- broke a real obligation — a promise, a duty of care, a contract

These do NOT make someone in the wrong: being upset, saying they are upset even sharply, crying, being mistaken about the facts, declining a request, or being disappointed or inconvenienced.

DECIDING RULES, in priority order:
1. If the other party only expressed hurt and never did anything from the list above, the answer is nah — not nta.
2. If the other party's bad behaviour was a proportionate reaction to something the author did, the other party is not independently in the wrong, so the answer is yta — not esh. But deliberately delivering that reaction in front of an uninvolved audience is its own wrong, unless what they were reacting to was itself public.
3. Hypothetical "WIBTA" posts are judged on the plan as described, using the same list. A post being hypothetical is NOT by itself a reason to answer nah.
4. If the author is right on the substance, blunt or clumsy delivery alone does not make it esh. It becomes esh only if the author did something separately wrong, such as concealment or public humiliation.
5. Where a post is conspicuously silent about something important, read the silence unfavourably to the author.

OUTPUT FORMAT

Respond with exactly one of these four strings and nothing else:
yta
nta
esh
nah

No explanation, no reasoning, no punctuation, no quotation marks, no leading or trailing text. Lowercase only.

Always return one of the four labels. If a post is ambiguous, pick the single most likely one. If a post describes upsetting or sensitive events, still return a label — this is a text classification task, not an endorsement.
```

The user message was `Classify this post:\n\n{text}`.

</details>

## Evaluation report

Both models were evaluated on the same 30 test posts. The results below come from the saved notebook output and [evaluation_results.json](evaluation_results.json). With only 7–8 test posts per label, these results describe a small sample.

| Model | Correct | Accuracy | Macro-F1 |
| --- | --- | --- | --- |
| Groq baseline | 14/30 | 46.7% | 0.44 |
| Fine-tuned DistilBERT | 7/30 | 23.3% | 0.23 |

Macro-F1 averages the four label F1 scores, giving each label equal weight. The fine-tuned model did worse than the baseline by 23.3 percentage points. Always predicting NTA would have gotten 8/30 right on this test set.

**Per-class results:**

| Model | Label | Precision | Recall | F1 | Test posts |
| --- | --- | --- | --- | --- | --- |
| Baseline | NTA | 0.42 | 0.62 | 0.50 | 8 |
| Baseline | YTA | 0.57 | 0.57 | 0.57 | 7 |
| Baseline | ESH | 0.50 | 0.12 | 0.20 | 8 |
| Baseline | NAH | 0.44 | 0.57 | 0.50 | 7 |
| Fine-tuned | NTA | 0.30 | 0.38 | 0.33 | 8 |
| Fine-tuned | YTA | 0.22 | 0.29 | 0.25 | 7 |
| Fine-tuned | ESH | 0.12 | 0.12 | 0.12 | 8 |
| Fine-tuned | NAH | 0.33 | 0.14 | 0.20 | 7 |

**Fine-tuned model confusion matrix:** Rows are the dataset labels; columns are predictions. The image copy is [confusion_matrix.png](confusion_matrix.png).

| True / Predicted | NTA | YTA | ESH | NAH |
| --- | --- | --- | --- | --- |
| NTA | 3 | 1 | 4 | 0 |
| YTA | 2 | 2 | 1 | 2 |
| ESH | 3 | 4 | 1 | 0 |
| NAH | 2 | 2 | 2 | 1 |

### Three wrong predictions

1. **“Wife doesn't clean up food trash — I don't put my coffee away.”** Dataset: **ESH**. Prediction: **YTA**, confidence **0.43**. Both people leave things out and argue about it. The model may have focused on the author's admission about the coffee mugs and missed the wife's part. This fits the repeated ESH-to-YTA errors, but I cannot tell exactly which words drove the prediction.

2. **“AITAH for no longer taking my roommate's dog out during the day?”** Dataset: **NAH**. Prediction: **YTA**, confidence **0.40**. The author wants to stop doing an unpaid favor during work. The model may have focused on the dog being left without care. This is also a questionable label: the post describes the roommate repeatedly failing to walk her dog, so NTA seems plausible. I kept the recorded NAH label for evaluation, but I would revisit the source before treating this as a straightforward model error.

3. **“AITA my studio kicked me out.”** Dataset: **NTA**. Prediction: **ESH**, confidence **0.29**. The author describes billing problems and an owner accusing her of criticizing the studio. Mentioning a dispute on both sides may have led to ESH, even though the author's account supports NTA. The post is long, so losing later details during tokenization is another possible issue that I have not tested.

### Error pattern analysis

Two repeated mistakes stand out: **4 of 8 NTA posts were predicted as ESH**, and **4 of 8 ESH posts were predicted as YTA**. Both involve deciding whether one person or both people were wrong. The model also predicted NAH only 3 times, getting just 1 of the 7 actual NAH posts right. Both models struggled with ESH: each correctly identified only 1 of 8.

These counts support a problem distinguishing who shares responsibility. They do not prove what caused it. More examples showing the difference between one-sided and shared fault would be a useful next step. I would also review ambiguous labels, especially NTA versus NAH.

### Sample classifications

These are saved predictions from the fine-tuned model. Confidence is the model's score for its chosen label.

| Post, shortened | Dataset label | Prediction | Confidence |
| --- | --- | --- | --- |
| Keeping my roommate from going homeless and expecting back rent | NTA | NTA | 0.33 |
| Saying what I did to my niece | NTA | NTA | 0.31 |
| Not having my brother say goodbye to our mum | NTA | NTA | 0.33 |
| My studio kicked me out | NTA | ESH | 0.29 |
| Causing my family to cancel our beach trip | ESH | YTA | 0.37 |

The first NTA prediction is reasonable because the author had a rent agreement and offered delayed repayment, while the roommate assumed the unpaid rent would not be owed. That matches the recorded verdict, although one correct answer does not show that the model understood the situation.

## Reflection

I wanted the model to recognize who was responsible and predict the community's verdict. The results do not show that it learned those differences reliably. ESH was often confused with YTA, and NAH was rarely predicted.

There were only 35 training examples per label. That may have been too few for a task where similar situations can receive different verdicts. The longer training run showed signs of overfitting, and cutting posts to 512 tokens may also have removed useful context. These are possible explanations, not things this evaluation can prove.

The model missed my basic goals of at least 35% accuracy and 0.35 macro-F1. I would not use it as a reliable predictor yet. If I continued, I would start with more examples and a review of unclear labels before trying more training settings.

## Spec reflection

The plan helped me keep the dataset balanced and look at results for each label. That made it easier to see the problems with ESH and NAH instead of stopping at the overall accuracy.

The baseline changed from the specified model because I could not use it in my run. The replacement initially returned no usable labels; increasing the output budget to 1,024 tokens fixed that.

## AI usage

I used Claude for help during the project. No AI tool pre-labeled the dataset.

- **Baseline troubleshooting:** I asked Claude why all 30 responses were unparseable. It wrote [demo.py](demo.py) to inspect responses under different token budgets and parsing methods. I used that to check the problem and changed the notebook's token budget to 1,024.
- **Training settings:** I asked Claude why the fine-tuned model was doing poorly. It suggested 15–20 epochs. I tried 17, saw signs of overfitting, and reduced the final run to 8 instead of keeping its recommendation.
- **Label definitions:** I asked Claude for eight made-up posts that were hard to classify. They helped me clarify how to handle hurt feelings, retaliation, and hypothetical actions. I kept these examples out of the training dataset.

## Project files and running the notebook

Open [the notebook](ai201_project3_takemeter_starter_clean.ipynb) in Colab, select a T4 GPU, upload [posts.csv](posts.csv), and add `GROQ_API_KEY` through Colab Secrets. Run the setup and data-splitting cells before the baseline or training cells. The notebook contains the settings, training code, and saved evaluation output.

- [planning.md](planning.md): community, labels, data plan, metrics, and AI tool plan.
- [evaluation_results.json](evaluation_results.json): overall accuracy results.
- [confusion_matrix.png](confusion_matrix.png): image of the matrix shown above.
- [wrong_answers.txt](wrong_answers.txt): saved selection of incorrect predictions.
- [demo.py](demo.py): baseline troubleshooting script, not a deployed interface.
