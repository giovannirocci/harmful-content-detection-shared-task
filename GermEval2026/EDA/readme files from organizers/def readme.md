# Detection of defamatory offences

This subdirectory contains the data for the fourth subtask of the shared task on “Harmful Content Detection in Social Media” within GermEval 2026: **the binary detection of defamatory offences** (i.e., Sections 185-187 of the German Criminal Code (StGB))

## Data annotation

The dataset contains all tweets for which a majority of the three annotators decided whether or not a statement in a tweet constitutes a defamatory offence under the German Criminal Code (StGB). Specifically, it must be checked whether any of the offences under Sections 185 to 187 of the German Criminal Code (StGB) have been committed. For information on the annotation of the data set and the elements of the offence, please refer to the work of [Zufall et al. (2019)](https://aclanthology.org/N19-1135/).

Origin and structure of the data

The dataset consists of posts and comments from a right-wing extremist movement from 12/12/2014 to 07/07/2016. The training data is provided as a CSV file. An entry has the following format:

“id”;“description”;“def”<br />
“1064396393598783”;“Oliver, ich guck doch schon mindestens einmal die Woche RTL2-NEWS.”;“false”

The test data is also distributed as a CSV file, containing an ID and the tweet text:
“id”;“description.”

## Anonymization of data

To anonymise the data, the mentions in the data set (training and test data) were replaced as follows:

* mentions of the press/press offices/news portals: [@PRE]
* mentions of the police/police authorities: [@POL]
* mentions of groups/organisations/associations: [@GRP]
* mentions of individuals: [@IND]

For example, the mentions of the organisation Greenpeace and the TV channel ARD in the following (fictitious) tweet would be replaced as follows: @greenpeace_de Euch liegt bei euren Aktionen wohl etwas an Sicherheit. Da muss man sich ja nur die letzte Doku ansehen, um das zu merken *@ARDde* => *[@GRP]* Euch liegt bei euren Aktionen wohl etwas an Sicherheit. Da muss man sich ja nur die letzte Doku ansehen, um das zu merken *[@PRE]*

No further preprocessing steps were performed on the data.

Files

* `def_trial.csv`: Sample of the training data set consisting of approximately 3,00 tweets that have been available since the trial phase to familiarise yourself with the data set.
