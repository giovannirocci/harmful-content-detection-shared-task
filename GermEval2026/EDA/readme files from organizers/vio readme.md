# Violence Detection

This subdirectory contains the data for the third subtask of the shared task on "Harmful Content Detection in Social Media" in the context of GermEval 2026: the **the fine-grained classification of disturbing statements about violence**. 

## Data annotation

The dataset contains all tweets for which the three to four annotators reached a majority decision regarding the type of violent statement. Specifically, a detailed annotation was made in five subtypes of violent statements and one negative category (i.e. a total of six categories):

- **nothing** (no violent expression), i.e., no negative statements about violence at all
- **propensity** (willingness to commit violence), i.e., the will or desire to use violence oneself
- **call2Violence** (call to violence), i.e., inciting or calling on other people to commit a violent act. 
- **support** (Endorsement of Violence), i.e., positive approval of violence/a violent event 
- **glorification** (Glorification of Violence), i.e., violence is presented as something particularly glorious and not just supported 
- **other**, i.e., other forms of worrying, violence-related statements 

## Origin and structure of the data 

The data set consists predominantly of posts and comments from a right-wing extremist movement from 12/12/2014 to 07/07/2016. A total of 10,479 tweets were annotated for violence detection, which were divided into training and test data using stratified sampling at a ratio of 85:15. The **training data** comprises 8,908 tweets. The training data is provided as a CSV file. An entry has the following format: 

"id";"description";"vio"<br />
"1064396393598783";"Oliver, ich guck doch schon mindestens einmal die Woche RTL2-NEWS.";"nothing"

The **test dataset** contains 1,571 tweets. It is identical to the GermEval 2025 test set to allow direct comparability between editions. The test data is also distributed as a CSV file, containing an ID and the tweet text:
"id";"description"<br />

## Anonymization of data

To anonymise the mentions in the data set (training and test data) were replaced as follows:
- mentions of the press/press offices/news portals: [@PRE]
- mentions of the police/police authorities: [@POL]
- mentions of groups/organisations/associations: [@GRP]
- mentions of individuals: [@IND]

For example, the mentions of the organisation Greenpeace and the TV channel ARD in the following (fictitious) tweet would be replaced as follows:
*@greenpeace_de* Euch liegt bei euren Aktionen wohl etwas an Sicherheit. Da muss man sich ja nur die letzte Doku ansehen, um das zu merken *@ARDde* => *[@GRP]* Euch liegt bei euren Aktionen wohl etwas an Sicherheit. Da muss man sich ja nur die letzte Doku ansehen, um das zu merken *[@PRE]*

No further preprocessing steps were performed on the data.

## Files

-  `vio_trial.csv`: Sample of the training data set consisting of approximately 1,000 tweets that have been available since the trial phase to familiarise yourself with the data set. 
