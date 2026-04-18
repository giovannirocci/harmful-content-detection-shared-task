# Additional Annotation

In addition to the annotations required for this shared task, the dataset was annotated for **hate speech**, **toxicity**, **targets**, and **emotions** to support the classification tasks. The additional annotations can be used, for example, as features or in the context of multi-task learning.

## Data

The dataset contains the additional annotations for the tweets in the training and test data for all subtasks. It is provided as a CSV file containing an ID and the gold standard for the additional annotations, reflecting the majority decision of three to four annotators. If an absolute majority could not be reached, the label is `"pari"`.

The ID can be used to link to the training or test data for each subtask and to add additional annotations to the tweets. An entry in this dataset has the following format:

```
"id";"HS";"TAR";"TOX";"EMO";"ANGER";"FEAR";"SURPRISE";"SADNESS";"JOY";"DISGUST";"ENVY";"JEALOUSY";"OTHER"
```

---

### Hate Speech (HS)

Hate speech is defined in line with the definition of the United Nations as:

> *Expressions that attack, demean, or discriminate against individuals or groups based on attributed group characteristics.*

These characteristics may include religion, ethnicity, nationality, race, skin color, descent, or gender.

**Possible annotations:**

| Value | Description |
|---|---|
| `TRUE` | The tweet contains hate speech. |
| `FALSE` | The tweet does not contain hate speech. |
| `don't know` | The majority of annotators stated that they were unable to determine whether the text constituted hate speech or not (e.g., due to a lack of contextual information). |

> **Note:** The presence of offensive or vulgar language alone is not sufficient for a tweet to be labeled as hate speech. The content must target a person or group based on protected or identity-related characteristics.

---

### Target Detection (TAR)

Each tweet was annotated to indicate its addressee or intended audience.

**Possible annotations:**

| Value | Description |
|---|---|
| `individual` | The tweet addresses one or more specific, identifiable persons (e.g., by name or direct mention). |
| `group` | The tweet addresses a collective, community, or institution (e.g., a government body), rather than specific individuals. |
| `public` | The tweet addresses the general public or has no clearly identifiable addressee. |

> **Note:** The annotation took the entire text of the tweet into account. Mentions (`@UserXY`) were of secondary importance.

---

### Toxicity Detection (TOX)

Toxic tweets are those that "poison" a conversation, for example, because they are rude, disrespectful, or inappropriate.

Toxicity was rated on a scale from **1** (not toxic) to **5** (very toxic). A tweet is considered more toxic the more it provokes aggressive reactions or causes participants to leave the conversation.

---

### Emotion Detection (EMO)

The last ten categories identify whether and which emotions are expressed in a tweet. The annotation followed a hierarchical two-step scheme:

#### 1. Binary Emotion Annotation (EMO)

Each tweet was labeled as either:

| Value | Description |
|---|---|
| `yes` | The tweet expresses at least one emotion. |
| `no` | The tweet is emotionally neutral. |

#### 2. Multi-Label Emotion Annotation

Tweets labeled as `yes` were further annotated with one or more emotion labels (multi-label setting). For emotionally neutral tweets (`EMO: "no"`), no further annotation of the specific emotion was performed — all remaining nine categories contain the value `NA` for these tweets.

**Possible sub-emotion labels:** `ANGER`, `FEAR`, `SURPRISE`, `SADNESS`, `JOY`, `DISGUST`, `ENVY`, `JEALOUSY`, `OTHER`

The six basic emotions (anger, fear, surprise, sadness, joy, disgust) follow the model of [Paul Ekman and Wallace V. Friesen (1971)](https://www.paulekman.com/wp-content/uploads/2013/07/Universals-And-Cultural-Differences-In-Facial-Expressions-Of.pdf). Additional categories include envy (resentment toward others' possessions or success), jealousy (fear of losing affection or advantage to another person), and other for emotions outside the predefined set.

**Possible annotations for each sub-emotion:**

| Value | Description |
|---|---|
| `TRUE` | The emotion (e.g., `ANGER`, `SADNESS`) is expressed in the tweet. |
| `FALSE` | The emotion is not expressed. |

**Example** — the following entry conveys both `ANGER` and `JOY`:

```
"id";"HS";"TAR";"TOX";"EMO";"ANGER";"FEAR";"SURPRISE";"SADNESS";"JOY";"DISGUST";"ENVY";"JEALOUSY";"OTHER"
951885771516513;TRUE;pari;5;yes;TRUE;FALSE;FALSE;FALSE;TRUE;FALSE;FALSE;FALSE;FALSE
```

> **Note:** It is possible for a tweet to be classified as emotional (`EMO = TRUE`) even though all sub-emotion categories have a value of `FALSE`. This outcome occurs because the overall emotion (`EMO`) and each sub-emotion category were determined independently of one another by majority vote. It can therefore happen that while the majority of annotators agree that the tweet contains any emotion, the majority voted `FALSE` for each individual sub-emotion because no specific emotion clearly dominated.
