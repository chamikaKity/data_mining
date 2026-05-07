National Institute of Business Management
Master of Science in Data Science
NIB2001CEM– Data Mining
Semester 2 - 2026
Course Work
The coursework comprises two parts.


##Part 1: Market Basket Analysis.

Description
One of the research activities in the area of association rule analysis is on development of efficient and scalable
mining algorithms. Association rule learning is a method for discovering interesting relations between variables in
large/big databases. It is intended to identify strong rules discovered in databases using some measures of
interestingness. Such information can be used as the basis for decisions about marketing activities such as, e.g.,
promotional pricing or product placements. Association rules mining is employed today in many application areas
including Web usage mining, intrusion detection, Continuous production, and bioinformatics.
Your task for this part is to identify and perform an association rule mining task. This involves
1. Preparing and preprocessing the data
2. Finding rules, including appropriate parameter setting
3. Determining which of the resulting rules are interesting
4. Figuring out how the interesting rules could be useful

Data Set:
Use the Online Retail data set (Online Retail.xlsx) described below: This is a transactional data set
which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and
registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers
of the company are wholesalers.

Attributes:
InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
Description: Product (item) name. Nominal.
1Quantity: The quantities of each product (item) per transaction. Numeric.
InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.
UnitPrice: Unit price. Numeric, Product price per unit in sterling.
CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
Country: Country name. Nominal, the name of the country where each customer resides.

The dataset is available Here


The Assignment report should contain the following:

1. Objectives: What are the potential benefits to be derived from association rule mining for this
domain. This is high level - not find patterns, but what would improve because of the use of
the patterns.
[05 Marks]

2. Data set description: What preprocessing was done to make it amenable for association rule
mining. Where choices were made (e.g., parameter settings for discretization, or decisions to
ignore an attribute), describe your reasoning behind the choices.
[20 Marks]

3. Rule mining process: Parameter settings, choice of algorithm and the time required, purpose
of the repetitions and the feedback found from the previous cycle.
[10 Marks]

4. Resulting rules: Summary (number of rules, general description), and a selection of those you
would show to a client.
[10 Marks]

5. Recommendations: What should the client do because of the rules discovered?
[05 Marks]


##Part 2: Clustering

In this assignment, we consider a set of observations on a number of red and white wine varieties involving their chemical
properties and ranking by tasters. Wine industry shows a recent growth spurt as social drinking is on the rise. The price of
wine depends on a rather abstract concept of wine appreciation by wine tasters. Pricing of wine depends on such a volatile
factor to some extent. Another key factor in wine certification and quality assessment is physicochemical tests which are
laboratory-based and takes into account factors like acidity, pH level, presence of sugar and other chemical properties. For
the wine market, it would be of interest if human quality of tasting can be related to the chemical properties of wine so that
certification and quality assessment and assurance process is more controlled.
Two datasets are available of which one dataset is on red wine and have 1599 different varieties and the other is on
white wine and has 4898 varieties (wine data.xlsx). All wines are produced in a particular area of Portugal. Data are collected
on 12 different properties of the wines one of which is Quality (i.e. the last column), based on sensory data, and the rest are
on chemical properties of the wines including density, acidity, alcohol content etc. All chemical properties of wines are
continuous variables. Quality is an ordinal variable with possible ranking from 1 (worst) to 10 (best). Each variety of wine is
tasted by three independent tasters and the final rank assigned is the median rank given by the tasters.

Description of attributes:
1. fixed acidity: most acids involved with wine or fixed or non-volatile (do not evaporate readily)
2. volatile acidity: the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste
3. citric acid: found in small quantities, citric acid can add ‘freshness’ and flavour to wines
4. residual sugar: the amount of sugar remaining after fermentation stops, it’s rare to find wines with less than 1
gram/litre and wines with greater than 45 grams/litre are considered sweet
5. chlorides: the amount of salt in the wine
6. free sulfur dioxide: the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite
ion; it prevents microbial growth and the oxidation of wine
7. total sulfur dioxide: amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine,
but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine
8. density: the density of water is close to that of water depending on the percent alcohol and sugar content
9. pH: describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between
3-4 on the pH scale
10. sulphates: a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and
antioxidant
11. alcohol: the percent alcohol content of the wine
12. Output variable (based on sensory data): quality (score between 0 and 10)Output variable (based on sensory data): quality (score between 0 and 10)

In this assignment you need to use the first 11 attributes to your calculations.


1st Objective:
Merge the two excel sheets (red and white) into one. Create a code in R Studio/python to conduct the k-means clustering
analysis of this combined sheet and try to distinguish red and white wines. Obviously, the number of clusters in this objective
is equal to 2. Plot a table with the results and produce in your report a confusion matrix in order to provide information regarding
classification accuracy and sensitivity (Hint: you know which sample belongs to each category). Before conducting the k-
means, please investigate if you need to add in your code any pre-processing task (justify your answer).
[10 Marks]

2nd Objective (only white wines):
You need to conduct the k-means clustering analysis of the white wine sheet. Find the ideal number of clusters (please justify
your answer). Choose the best two possible numbers of clusters and perform the k-means algorithm for both candidates.
Validate which clustering test is more accurate. For the winning test, get the mean of the each attribute of each group. Before
conducting the k-means, please investigate if you need to add in your code any pre-processing task (justify your answer).
Write a code in R Studio/python to address all the above issues. In your report, check the consistency of those produced
clusters, with information obtained from column 12.
[20 Marks]

3rd Objective (only white wines):
You need to conduct the hierarchical clustering (agglomerative) analysis of the white wine sheet. Investigate the hclust()
function for single, complete, average methods. Create the visualization of all methods using a dendrogram. Look at the
cophenetic correlation between each clustering result using cor.dendlist. Discuss the produced results after using the coorplot
function. Write a code in R Studio/python to address all the above issues.
[20 Marks]
