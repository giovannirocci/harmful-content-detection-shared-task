# Detection of attacks on the liberal basic democratic order

This subdirectory contains the data for the second subtask of the shared task on "Harmful Content Detection in Social Media" in the context of GermEval 2026: the **fine-grained detection of various attacks on the libral democratic basic order of the Federal Republic of Germany**. 

## Data annotation

The data set contains all tweets for which the three to four annotators could reach a majority decision regarding the form of attack. Specifically, the annotators assigned the tweets to one of the following classes: 
- **subversive:** A will is expressed to forcibly remove the existing government and overthrow it (e.g., through militant action, disruption of the power grid, etc.).   
- **agitation:** Agitative efforts are expressed. That includes the announcement of actions such as the dissemination of propaganda material of unconstitutional and terrorist organisations or the damaging of state symbols such as the flag of the Federal Republic of Germany.   
- **criticism:** Tweets in which legitimate criticism of the government, officials, government employees, authorities or parties was expressed were assigned to this class. 
- **nothing:** Tweets in this category contain neither criticism nor an attack against the free democratic basic order. However, neutral or positive statements on government decisions can be expressed in the tweets. 

## Origin and structure of the data 

The **training data** for GermEval 2025 has been expanded and now includes a total of 16,256 tweets. The data set consists predominantly of posts and comments from a right-wing extremist movement from 12/12/2014 to 07/07/2016. The training data is provided as a CSV file. An entry has the following format: 

"id";"description";"dbo"<br />
"1064396393598783";"Oliver, ich guck doch schon mindestens einmal die Woche RTL2-NEWS.";"nothing"

The **test dataset** contains 3,194 tweets. It is identical to the GermEval 2025 test set to allow direct comparability between editions. The test data is also distributed as a CSV file, containing an ID and the tweet text:
"id";"description"<br />

## Anonymization of data

To anonymise the mentions in the data set were replaced as follows:
- mentions of the press/press offices/news portals: [@PRE]
- mentions of the police/police authorities: [@POL]
- mentions of groups/organisations/associations: [@GRP]
- mentions of individuals: [@IND]

For example, the mentions of the organisation Greenpeace and the TV channel ARD in the following (fictitious) tweet would be replaced as follows:
*@greenpeace_de* Euch liegt bei euren Aktionen wohl etwas an Sicherheit. Da muss man sich ja nur die letzte Doku ansehen, um das zu merken *@ARDde* => *[@GRP]* Euch liegt bei euren Aktionen wohl etwas an Sicherheit. Da muss man sich ja nur die letzte Doku ansehen, um das zu merken *[@PRE]*

No further preprocessing steps were performed on the data.

## Files

-  `dbo_trial.csv`: Sample of the training data set consisting of approximately 1,000 tweets that have been available since the trial phase to familiarise yourself with the data set. 
