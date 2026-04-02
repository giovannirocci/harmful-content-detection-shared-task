# Harmful Content Detection
## A GermEval 2026 Shared Task 

This repository contains the annotated Twitter dataset provided for the shared task in the context of GermEval 2026. The shared task is organised into the following three subtasks:

1. The **binary detection** of so-called **calls to actions**, i.e., calls for risky actions (e.g., criminal offences, demonstrations with possible escalation potential)
2. The **fine-grained classification** into four forms of statements and (violent) attacks against the **liberal democratic basic order** of the Federal Republic of Germany
3. The **fine-grained classification** into six different categories of **violence-related** statements in tweets
4. The **binary detection** of **defamatory offences** (i.e., Sections 185-187 of the German Criminal Code (StGB))

For GermEval 2026, a sample of around **22,200 tweets** from a total corpus of approximately 800,000 tweets was annotated. The data sets for **subtasks 1–3** are based on data already annotated in the previous edition and were expanded for GermEval 2026.

The annotation of the first three subtasks was carried out by members of Mittweida University of Applied Sciences. Each tweet was annotated by three to four annotators. Only tweets for which there was a majority decision by the three to four annotators were included in the final data sets. This resulted in training datasets comprising approximately **15,000-16,000** tweets.

The particularly challenging annotation of the fourth subtask was carried out by staff at Mittweida University of Applied Sciences and the Central Office for Information Technology in the Security Sector (ZITiS), both of whom have extensive practical experience in evaluating harmful and criminally relevant content. The annotation of the fourth subtask will be completed in mid/late February.

The data sets for each subtask, and further explanations of the data, can be found in the repository’s individual subdirectories. 

**In addition to the annotations required for this shared task, the data set was annotated for hate speech, toxicity, target and emotions to support
the classification tasks** (e.g. in the sense of multi-task learning or as a feature). A data set containing the tweets of all subtasks with the additional annotations is provided in a separate subfolder.

## Take Part in the Shared Task

To take part in this competition, please register [here](https://www.codabench.org/competitions/edit/14006/#/).

**Important Deadlines:**

| <u>**Date**</u> |  | <u>**Phase/Deadline**</u> |
| :------------- | :------------- | :------------- |
| <span style='color: red;'>21 February - 16 March 2026</span> |  | Trial phase |
| 10 April - 16 May 2026 |  | Training phase |
| 24 March 2026 |  | Baseline model ready |
| 17 May - 21 June 2026 |  | Competition phase |
| 15 July 2026 |  | Paper submission due |
| 15 August 2026 |  | Camera ready due |
