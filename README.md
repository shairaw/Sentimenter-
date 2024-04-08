<h1>Sentimeter -Sentiment Analysis for Film Reviews</h1>

 ### [YouTube Demonstration](https://youtu.be/7eJexJVCqJo)

<h2>Description</h2>
The product is a user-friendly interface in the form of a GUI with a Logistic regression model at the backend. The team employed Scrum Agile Methodology for structured management, the project's success lies in its efficient transformer model development, achieving a commendable 90% accuracy. 
<br />


<h2>Languages and Utilities Used</h2>

- <b>Python</b> 
- <b>Pickle</b>
- <b>Jupyter Notebook</b>

<h2>Environments Used </h2>

- <b>macOS</b> (Catalina)

<h2>Program walk-through:</h2>

# Introduction

In today’s contemporary digital landscape, companies are trying to minimise their costs and automate digital tasks in order to maximise their profits and growth. In this pursuit, making inferences from textural data is a key task for example determining the sentiment of reviews. Our project is to automate the sentiment analysis of movie reviews distinguishing them as positive or negative using machine learning. Sentiment analysis (SA), otherwise known as opinion mining, is the field of research that involves processing and analysing opinions, behaviours, and sentiments of people towards specific issues, events, organisations, products, services, or their respective attributes [1]. The importance of reviews cannot be understated [2]. Unlike traditional methods such as focus groups or interviews, customer reviews represent a dynamic and real-time source of insights into product performance, customer sentiments, and prevailing attitudes, comprehensively capturing large amounts of user feedback, while being much less costly to gather and analyse.
The central challenges that companies are facing at the moment are as follows. Firstly, we observe the high labour costs of employees who are reading reviews and classifying them manually which has reduced profit margins for companies, especially in the film industry. Wasting valuable time and money in this way is avoidable through automation taking over repetitive tasks. Secondly, user engagement is low with websites which is impacting ad revenue and causing many customers to shift to other platforms. Lastly, classification by human employees is prone to possible biases and limitations. There needs to be a uniform, data-driven approach to this for more accurate insights.

We propose a solution to these issues in the form of a compact, user-friendly Graphs User Interface (GUI) which encompasses an automated review classification system. The automation model uses an accurate classification machine learning model trained on existing movie reviews. This will allow companies using the product to make in- formed and data-driven decisions to allocate resources correctly, increasing their cost- to-knowledge ratio about user preferences about movies. This is vital to maximising profits. The user-friendly GUI will facilitate fewer employees to quickly and easily find the sentiments for larger numbers of reviews with little to no training required. This report discusses our methodology for data acquisition, preprocessing, training, testing, and identification of the best model, as well as the integration of a GUI. We have also discussed the challenges of natural language processing including a wide range of review lengths, structures, and expressions as well as which methods helped us to address these problems. Automating this task reduces the time it takes to classify large amounts of reviews, thus increasing the efficiency and accuracy of sentiment analysis.
## Applications of Sentiment Analysis

Sentiment Analysis finds its application in a variety of domains.

### A. Online Commerce

The most general use of sentiment analysis is in e-commerce activities. Websites allow their users to submit their experience about shopping and product qualities. They provide a summary for the product and different features of the product by assigning ratings or scores. Customers can easily view opinions and recommendation information on the whole product as well as specific product features. Graphical summary of the overall product and its features is presented to users. Popular merchant websites like [Amazon](https://www.amazon.com) provide reviews from editors and also from customers with rating information. [TripAdvisor](http://tripadvisor.in) is a popular website that provides reviews on hotels, travel destinations. They contain 75 million opinions and reviews worldwide. Sentiment analysis helps such websites by converting dissatisfied customers into promoters by analyzing this huge volume of opinions.

### B. Voice of the Market (VOM)

Voice of the Market is about determining what customers are feeling about products or services of competitors. Accurate and timely information from the Voice of the Market helps in gaining competitive advantage and new product development. Detection of such information as early as possible helps in direct and target key marketing campaigns. Sentiment Analysis helps corporate to get customer opinion in real-time. This real-time information helps them to design new marketing strategies, improve product features and can predict chances of product failure.

### C. Voice of the Customer (VOC)

Voice of the Customer is concern about what individual customer is saying about products or services. It means analyzing the reviews and feedback of the customers. VOC is a key element of Customer Experience Management. VOC helps in identifying new opportunities for product inventions. Extracting customer opinions also helps identify functional requirements of the products and some non-functional requirements like performance and cost.

### D. Brand Reputation Management

Brand Reputation Management is concern about managing your reputation in market. Opinions from customers or any other parties can damage or enhance your reputation. Brand Reputation Management (BRM) is a product and company focused rather than customer. Now, one-to-many conversations are taking place online at a high rate. That creates opportunities for organizations to manage and strengthen brand reputation. Now Brand perception is determined not only by advertising, public relations and corporate messaging. Brands are now a sum of the conversations about them. Sentiment analysis helps in determining how company’s brand, product or service is being perceived by community online.

### E. Government

Sentiment analysis helps government in assessing their strength and weaknesses by analyzing opinions from public. For example, “If this is the state, how do you expect truth to come out? The MP who is investigating 2g scam himself is deeply corrupt.”. this example clearly shows negative sentiment about government. Whether it is tracking citizens’ opinions on a new 108 system, identifying strengths and weaknesses in a recruitment campaign in government job, assessing success of electronic submission of tax returns, or many other areas, we can see the potential for sentiment analysis.


# Data Pre-processing 

Preprocessing ensures that the data is ready for analysis. The following steps reduce noise in the data and format the dataset such that it is standardised. This promotes a more consistent and understandable dataset that can be used to better increase the performance of the machine-learning model. In this stage, we performed the following steps to clean the dataset and remove any unwanted or irrelevant information:
1. Removed rows that include missing data
2. Removed HTML tags that are present in the movie reviews
3. Remove characters that do not carry sentiment: special characters, punctuation (excluding ’!’ and ’?’), numbers
4. Remove stopwords: common words such as ’the’, ’and’ ’is’ are ignored as they provide negligible sentiment
5. Split the movie reviews into individual tokens
   
In our methodology, we have opted to not perform stemming or lemmatisation ofwords, as literature suggests that these methods decrease the sentiment within the text. After separating the movie reviews into sentiment classes of either positive or neg- ative by rating, we shuffled the dataset to ensure a random order of data because we wanted to avoid any potential bias that could arise in the sorting of the data.
Next, we applied feature extraction, feature models play a very important role in classification purposes, this refers to an approach which defines in which feature and what way they are going to be used to classify new data into the specific type of class. In this context, it means the selection of text and converting it into scalable vectors of numbers. The current literature suggests that the choice of feature extraction methods depends on the suitability of the classification model. Hence, we have chosen to use N-Grams with TF-IDF weighting as the literature suggests that this method achieves the best performance.
To avoid overfitting the machine learning model, which is when the model fails to generalise to unseen data, we have performed a train-test split on the dataset. Literature suggests that an 80:20 train-test split is a good ratio that promotes goodness of fit. Then, we trained the model on the training subset and tested it on the (unseen) test set to validate that it can perform well on new data. Finally, we can assess the model’s performance on the unseen test set, to get an unbiased estimate of the accuracy.

# Testing and comparing Models

After rigorous testing and re-testing of multiple models, Logistic Regression emerged as the optimal choice due to its simplicity, ease of training, and low computational requirements. Particularly effective in low-dimensional datasets with sufficient training points, it proved less prone to overfitting. Addressing the major limitation of assuming linearity between the dependent and independent variables, Logistic Regression proved particularly effective in our task. Its suitability for low-dimensional datasets with am- ple training points minimises the risk of overfitting. Through meticulous training on a substantial 50,000 reviews dataset, this model demonstrated remarkable prowess, se- curing an impressive accuracy of 90%. Furthermore, the model only took 6 seconds to train on the 50,000 review dataset (attach reference to table) and could be expanded to include real-time reviews if need be, emerging as a viable solution if the project is expanded in the future. Recognising its efficiency, Logistic Regression was integrated into our GUI to enhance user accessibility and streamline the sentiment analysis process for end-users. This model, with its inherent strengths and adaptability, stands as the cornerstone solution within our project.

# GUI
Prioritising user experience, our GUI is designed with a keen focus on user-friendliness, ensuring an intuitive and straightforward interface for seamless interaction. Key fea- tures include labelled buttons, a minimalistic design approach, sensible font choices, and an efficient pop-up system for user confirmation and output display. Whilst designing the GUI, privacy considerations are integral. Standardising formats and minimising features not only contribute to a clean interface but also reduce the risk of unintentional data exposure. Stringent data privacy measures are implemented to safeguard user information throughout the sentiment analysis process. This GUI was implemented using Pickle library through Jupyter Notebook. 


# Privacy frameworks
The widespread adoption of sentiment analysis in deriving insights from movie reviews brings forth ethical considerations and potential misuse of this technology. These im- plications, particularly concerning privacy, have initiated debates and raised important concerns. Transparency in the methodologies and algorithms employed in sentiment analysis is paramount, especially when dealing with SVM models that, due to their complex kernels, can behave like black boxes. Ensuring interpretability is crucial to maintain accountability in the use of this technology. The model’s training process
11
prioritized ethical considerations by utilizing open-source datasets available on Kaggle. This ensures the availability of clear documentation regarding data sources, user con- sent, and any licensing agreements. Notably, the dataset aligns with GDPR standards, guaranteeing compliance with regulations for obtaining and managing user consent.
The General Data Protection Regulation (GDPR) [12], enacted by the European Union, is a comprehensive framework for data protection and privacy. Should the project extend to real-time data collection, strict adherence to GDPR best practices will be followed. Any testing conducted will align with the GDPR framework, ensuring that user data is handled with utmost care and in full compliance with privacy regulations.
Privacy issues are an important aspect, especially in dealing with personal informa- tion that might expose individuals. Data storage practices, including anonymisation of users, have been implemented, addressing concerns related to the use of real names in either dataset.
While open datasets are valuable for training models, they may not be entirely representative of the broader population. This limitation introduces the risk of biased models that could perpetuate existing societal biases. The dataset used lacks informa- tion on the demographic or geographical location of users leaving comments, potentially leading to the underrepresentation of opinions from specific regions. Acknowledging and addressing these biases is crucial for the responsible and fair deployment of sentiment analysis models.
The project operates under the Creative Commons Attribution-4.0 International License (CC BY-SA 4.0) [13]. This license allows for the sharing, copying, and redis- tribution of the material in any medium or format for any purpose, even commercially. Additionally, it permits the remixing, transforming, and building upon the material for any purpose, under the condition that appropriate credit is given, a link to the license is provided, and changes made are indicated. This licensing framework guarantees the continued freedom to use and build upon our project.


# Future Research
Sentiment analysis has many uses beyond only determining the positive or negative content of reviews. It can be used in all phases of the filmmaking process, from pre- production market research to post-release audience engagement. The classifier can be used to fuel content recommendation systems, by recommending similar movies to the ones a user enjoys. Studios and filmmakers are using AI-based sentiment analysis to enhance their ad- vertising strategies. Using these insights companies are able to enhance personali- sations, offering tailored recommendations aligned with individual preferences. This targeted approach increases user satisfaction and content engagement. Another area of growth is the use of AI-based sentiment analysis for streaming platforms. By analyzing viewer sentiment data, streaming services can personalize recommendations and suggest content that is likely to be well-received by individual viewers.

Finally, AI-based sentiment analysis is being used to predict box office success. By analyzing viewer sentiment data related to upcoming releases, AI can provide insights into which films are likely to be successful and which may struggle at the box office. For instance, the company Vault51 used AI-based sentiment analysis to predict box office success for the movie ’Joker’. They analyzed social media data to predict the box office success and their predictions turned out to be accurate.


<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
